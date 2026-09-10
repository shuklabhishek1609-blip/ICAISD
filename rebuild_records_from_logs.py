"""Rebuild RunRecord JSONs from Kaggle run logs.

The real results/*.json files live inside Kaggle version outputs that proved
awkward to retrieve. Every field they hold is also printed to the notebook log,
so parse the log instead. Provenance goes in each record's `notes` field --
these are reconstructions, and nothing untraceable should reach the paper.

Only models whose run is protocol-valid are rebuilt. nlinear and stid were
trained at patience=15 by the stale notebook and are being re-run, so they are
deliberately absent; sttn has not run yet.

Usage:  python rebuild_records_from_logs.py <executed_notebook.ipynb>
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'results')

# Fields the log does not print. FLOPs come from the LaTeX table the run
# emitted; `flops_reliable` mirrors what count_flops decided at the time --
# hl's tracer saw nothing (parameter-free), so it printed '--'.
STATIC = {
    #          MFLOPs/sample, reliable
    'hl':     (None,    False),
    'lstm':   (1042.77, True),
    'stgcn':  (1018.76, True),
}

SUMMARY_RE = re.compile(
    r'MAE (?P<mae>[\d.]+) \| train (?P<kj>[\d.]+) kJ \((?P<ep>\d+) e(?:p|pochs)\)'
    r'(?: \| infer/1k (?P<infer>[\d.]+) J)? \| (?P<lat>[\d.]+) ms \| (?P<params>[\d,]+) params')


def _floats(text, pattern):
    return [float(m) for m in re.findall(pattern, text)]


def parse_model_block(block, name):
    """Pull one model's record out of its slice of the log."""
    rec = {'model': name, 'dataset': 'SD', 'seed': 2023,
           'gpu_name': 'Tesla T4', 'energy_method': 'nvml_total_energy'}

    avg = re.search(r'Average Test MAE: ([\d.]+), Test RMSE: ([\d.]+), Test MAPE: ([\d.]+)', block)
    if not avg:
        return None
    rec['mae'], rec['rmse'], rec['mape'] = (float(g) for g in avg.groups())

    horizons = re.findall(
        r'Horizon \d+, Test MAE: ([\d.]+), Test RMSE: ([\d.]+), Test MAPE: ([\d.]+)', block)
    rec['horizon_mae'] = [float(h[0]) for h in horizons]
    rec['horizon_rmse'] = [float(h[1]) for h in horizons]
    rec['horizon_mape'] = [float(h[2]) for h in horizons]

    # Total training energy is only logged for models that actually train;
    # Historical Last skips the loop and is a true zero, not a missing value.
    tot = re.search(r'Total training energy: ([\d.]+)J over (\d+) epochs', block)
    if tot:
        rec['train_joules'] = float(tot.group(1))
        rec['epochs_run'] = int(tot.group(2))
    else:
        rec['train_joules'] = 0.0
        rec['epochs_run'] = 0

    # Mean seconds per epoch, from the per-epoch Train Time stamps.
    epoch_s = _floats(block, r'Train Time: ([\d.]+)s/epoch')
    rec['epoch_seconds'] = sum(epoch_s) / len(epoch_s) if epoch_s else float('nan')
    rec['train_seconds'] = sum(epoch_s) if epoch_s else 0.0

    infer = re.search(r'Inference energy: ([\d.]+)J over (\d+) samples', block)
    rec['infer_joules_per_1k'] = (float(infer.group(1)) / int(infer.group(2)) * 1000
                                  if infer else float('nan'))

    summ = SUMMARY_RE.search(block)
    if summ:
        rec['latency_ms'] = float(summ.group('lat'))
        rec['params'] = int(summ.group('params').replace(',', ''))
    else:
        p = re.search(r'The number of parameters: (\d+)', block)
        rec['params'] = int(p.group(1)) if p else 0
        rec['latency_ms'] = float('nan')
    if rec['params'] == 1:
        rec['params'] = 0        # hl's dummy tensor; runner counts trainable only

    mflops, reliable = STATIC.get(name, (None, False))
    rec['flops_per_sample'] = mflops * 1e6 if mflops else float('nan')
    rec['flops_reliable'] = reliable

    rec['notes'] = ('RECONSTRUCTED from Kaggle notebook log, not the original '
                    'results JSON. All accuracy/energy/epoch fields are values '
                    'printed verbatim by the run; train_seconds and '
                    'epoch_seconds are derived from the per-epoch Train Time '
                    'stamps and so exclude validation; FLOPs come from the '
                    "run's own LaTeX table.")
    return rec


def notebook_log(path):
    with open(path, encoding='utf-8') as fh:
        nb = json.load(fh)
    out = []
    for cell in nb['cells']:
        for o in cell.get('outputs', []):
            if o.get('output_type') == 'stream':
                out.append(''.join(o.get('text', [])))
    return '\n'.join(out)


def split_blocks(log):
    """Slice the log at each model banner."""
    marks = [(m.start(), m.group(1)) for m in
             re.finditer(r'^(\w+)  seed=2023', log, re.M)]
    blocks = {}
    for i, (start, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(log)
        blocks[name] = log[start:end]
    return blocks


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    log = notebook_log(sys.argv[1])
    blocks = split_blocks(log)
    os.makedirs(OUT_DIR, exist_ok=True)

    written = []
    for name in STATIC:
        if name not in blocks:
            print(f'-- {name}: no block in this log, skipped')
            continue
        rec = parse_model_block(blocks[name], name)
        if rec is None:
            print(f'-- {name}: block found but no Average Test MAE, skipped')
            continue
        path = os.path.join(OUT_DIR, f'{name}_SD_2019_s2023.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, indent=2)
        written.append((name, rec))
        print(f'wrote {path}')

    print()
    for name, r in written:
        print(f'{name:8} MAE {r["mae"]:6.2f} | {r["train_joules"]/1000:8.1f} kJ | '
              f'{r["epochs_run"]:3d} ep | {r["latency_ms"]:7.2f} ms | '
              f'{r["params"]:>9,} params | infer/1k {r["infer_joules_per_1k"]:.3f} J')


if __name__ == '__main__':
    main()
