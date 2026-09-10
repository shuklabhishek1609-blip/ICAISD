"""Unified experiment driver.

LargeST's design is one main.py per model, each parsing its own argv. That is
fine from a shell but useless from a notebook, and it makes it easy for the
comparison to drift (different batch sizes, different input_dim). This module
rebuilds every model through one code path with one set of shared settings, so
the only thing varying across runs is the architecture.

Per-model hyperparameters are copied verbatim from the corresponding
experiments/*/main.py defaults -- we are not tuning anyone's baseline down.
"""

import gc
import os
import random

import numpy as np
import torch

from src.base.model import BaseModel
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx
from src.utils.logging import get_logger
from src.utils.metrics import masked_mae

from .engine import InstrumentedEngine
from .instrument import RunRecord, count_flops, measure_latency, gpu_name
from .models_lite import STID, NLinear, HistoricalLast


class Args:
    """Stand-in for the argparse namespace load_dataset expects."""

    def __init__(self, **kw):
        self.years = '2019'
        self.seq_len = 12
        self.horizon = 12
        self.input_dim = 3
        self.output_dim = 1
        self.bs = 64
        self.__dict__.update(kw)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# model builders -- each returns (model, optimizer_factory, scheduler_factory,
# clip_grad_value). Factories rather than instances so the optimiser is built
# after the model is on-device.
# --------------------------------------------------------------------------

def _build_hl(node_num, adj_path, args, device):
    model = HistoricalLast(node_num=node_num, input_dim=args.input_dim,
                           output_dim=args.output_dim)
    return model, None, None, 0


def _build_nlinear(node_num, adj_path, args, device):
    model = NLinear(node_num=node_num, input_dim=args.input_dim,
                    output_dim=args.output_dim, individual=True)
    opt = lambda m: torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    return model, opt, None, 5


def _build_stid(node_num, adj_path, args, device):
    model = STID(node_num=node_num, input_dim=args.input_dim,
                 output_dim=args.output_dim, embed_dim=32, node_dim=32,
                 temp_dim=32, layers=3, dropout=0.15)
    opt = lambda m: torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
    return model, opt, None, 5


def _build_lstm(node_num, adj_path, args, device):
    from src.models.lstm import LSTM
    model = LSTM(node_num=node_num, input_dim=args.input_dim,
                 output_dim=args.output_dim, init_dim=32, hid_dim=64,
                 end_dim=512, layer=2, dropout=0.1)
    opt = lambda m: torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    return model, opt, None, 5


def _build_stgcn(node_num, adj_path, args, device):
    from src.models.stgcn import STGCN
    Kt, Ks, block_num = 3, 3, 2

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    gso = normalize_adj_mx(adj_mx, 'scalap')[0]
    gso = torch.tensor(np.asarray(gso), dtype=torch.float32).to(device)

    Ko = args.seq_len - (Kt - 1) * 2 * block_num
    blocks = [[args.input_dim]]
    for _ in range(block_num):
        blocks.append([64, 16, 64])
    blocks.append([128] if Ko == 0 else [128, 128])
    blocks.append([args.horizon])

    model = STGCN(node_num=node_num, input_dim=args.input_dim,
                  output_dim=args.output_dim, gso=gso, blocks=blocks,
                  Kt=Kt, Ks=Ks, dropout=0.5)
    opt = lambda m: torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=10, gamma=0.95)
    return model, opt, sched, 0


def _build_gwnet(node_num, adj_path, args, device):
    from src.models.gwnet import GWNET
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, 'doubletransition')
    supports = [torch.tensor(np.asarray(a), dtype=torch.float32).to(device) for a in adj_mx]

    model = GWNET(node_num=node_num, input_dim=args.input_dim,
                  output_dim=args.output_dim, supports=supports, adp_adj=1,
                  dropout=0.3, residual_channels=32, dilation_channels=32,
                  skip_channels=256, end_channels=512)
    opt = lambda m: torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    return model, opt, None, 5


def _build_sttn(node_num, adj_path, args, device):
    from src.models.sttn import STTN
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, 'doubletransition')
    supports = [torch.tensor(np.asarray(a), dtype=torch.float32).to(device) for a in adj_mx]

    model = STTN(node_num=node_num, input_dim=args.input_dim,
                 output_dim=args.output_dim, device=device, supports=supports,
                 blocks=2, mlp_expand=2, hidden_channels=32, end_channels=512,
                 dropout=0.1)
    opt = lambda m: torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    return model, opt, None, 5


# STGODE is deliberately absent: it precomputes a DTW similarity matrix over
# every sensor pair (fastdtw, O(n^2) pairs), which is hours of CPU for SD=716
# before a single epoch runs. Note the omission in the paper rather than
# pretending the lineup is exhaustive. DCRNN/DGCRN/D2STGNN/AGCRN/ASTGCN/DSTAGNN
# are excluded for a different reason -- they need their own engine subclasses,
# so adding them means duplicating the instrumentation.
MODEL_REGISTRY = {
    'hl': _build_hl,
    'nlinear': _build_nlinear,
    'stid': _build_stid,
    'lstm': _build_lstm,
    'stgcn': _build_stgcn,
    'gwnet': _build_gwnet,
    'sttn': _build_sttn,
}


def run_experiment(model_name, dataset='SD', years='2019', seed=2023, bs=64,
                   max_epochs=100, patience=30, device=None, results_dir='results',
                   log_root='./logs'):
    """Train one model, measure everything, return a RunRecord.

    Patience matches LargeST's 30. An earlier version used 15 to fit a Colab
    session, but the observed runs showed 15 truncating STID -- the model this
    paper argues *for* -- as well as the expensive ones, so it was not the
    neutral shortcut it looked like. Every model in the table must share one
    value; do not lower it for a single run.
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{model_name}'; have {sorted(MODEL_REGISTRY)}")

    device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    set_seed(seed)

    args = Args(years=years, bs=bs)
    data_path, adj_path, node_num = get_dataset_info(dataset)

    log_dir = os.path.join(log_root, model_name, dataset)
    logger = get_logger(log_dir, f'{model_name}_{dataset}', f'record_s{seed}.log')
    logger.info(f'model={model_name} dataset={dataset} seed={seed} nodes={node_num}')

    dataloader, scaler = load_dataset(data_path, args, logger)

    model, opt_factory, sched_factory, clip = MODEL_REGISTRY[model_name](
        node_num, adj_path, args, device)
    model = model.to(device)

    optimizer = opt_factory(model) if opt_factory else torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = sched_factory(optimizer) if sched_factory else None

    engine = InstrumentedEngine(
        device=device, model=model, dataloader=dataloader, scaler=scaler,
        sampler=None, loss_fn=masked_mae, lrate=1e-3, optimizer=optimizer,
        scheduler=scheduler, clip_grad_value=clip, max_epochs=max_epochs,
        patience=patience, log_dir=log_dir, logger=logger, seed=seed)

    # Trainable parameters, not param_num(): Historical Last carries a dummy
    # tensor so the optimiser has something to hold, and counting it would put
    # "1 parameter" in a table row for a model that has none. Identical to
    # param_num() for every model that actually learns.
    trainable = sum(p.nelement() for p in model.parameters() if p.requires_grad)

    record = RunRecord(model=model_name, dataset=dataset, seed=seed,
                       params=trainable, gpu_name=gpu_name(),
                       energy_method=engine._meter.method)

    results = engine.train()
    record.mae = results['mae']
    record.rmse = results['rmse']
    record.mape = results['mape']
    record.horizon_mae = results['horizon_mae']
    record.horizon_rmse = results['horizon_rmse']
    record.horizon_mape = results['horizon_mape']

    record.train_joules = engine.train_joules
    record.train_seconds = engine.train_seconds
    record.epochs_run = engine.epochs_run
    record.epoch_seconds = float(np.mean(engine.epoch_seconds)) if engine.epoch_seconds else float('nan')
    record.infer_joules_per_1k = engine.measure_inference_energy()

    # static cost, measured on a single batch
    sample = torch.randn(bs, args.seq_len, node_num, args.input_dim, device=device)
    flops, reliable = count_flops(model, sample)
    record.flops_per_sample = flops / bs if flops == flops else float('nan')
    record.flops_reliable = reliable
    record.latency_ms = measure_latency(model, sample)

    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f'{model_name}_{dataset}_{years}_s{seed}.json')
    record.save(out)
    logger.info(f'saved {out}')

    del model, engine, dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return record
