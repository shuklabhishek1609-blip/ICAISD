"""Compute/energy instrumentation for the ICAISD 2026 traffic-forecasting study.

Three measurements underpin the paper's efficiency axis:

  * energy  -- real GPU energy draw (joules) over a code region, via NVML
  * flops   -- static forward-pass cost, via torch.utils.flop_counter
  * latency -- inference wall time, via CUDA events

Energy is the primary metric. FLOPs are reported as a hardware-independent
cross-check, but see `count_flops` for why they are not always trustworthy.
"""

import json
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict

import torch

try:
    import pynvml
    _NVML = True
except ImportError:  # pragma: no cover - depends on runtime
    _NVML = False


# Grid carbon intensity, gram CO2e per kWh.
#
# SOURCED 2026-09-09. LargeST is California data, so we use the EPA eGRID
# subregion CAMX (WECC California) total output CO2e rate: 430.0 lb/MWh from
# eGRID2023 Rev 2 (released 2025-06-12), converted at 0.45359237 kg/lb ->
# 195.04 gCO2e/kWh. The previous value was an uncited placeholder of 400.0,
# which overstated every carbon figure by 2.05x.
DEFAULT_GRID_INTENSITY = 195.04


# --------------------------------------------------------------------------
# energy
# --------------------------------------------------------------------------

class GPUEnergyMeter:
    """Measures GPU energy over a region.

    Prefers NVML's total-energy counter, which is a hardware accumulator and
    needs no sampling. That counter exists on Volta and newer, which covers
    every GPU Colab hands out (T4, L4, A100). Where it is missing we fall back
    to integrating instantaneous power on a background thread, which is
    noisier -- `self.method` records which path was used so the paper can say
    so honestly.
    """

    def __init__(self, device_index=0, sample_interval=0.05):
        self.device_index = device_index
        self.sample_interval = sample_interval
        self.method = "unavailable"
        self.joules = 0.0
        self._handle = None
        self._stop = None
        self._thread = None
        self._samples = []

        if not _NVML:
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"NVML init failed, energy will not be measured: {exc}")
            self._handle = None
            return

        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            self.method = "nvml_total_energy"
        except Exception:
            self.method = "power_integration"

    @property
    def available(self):
        return self._handle is not None

    def _read_total_mj(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)

    def _poll(self):
        while not self._stop.is_set():
            try:
                milliwatts = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self._samples.append((time.time(), milliwatts / 1000.0))
            except Exception:
                pass
            self._stop.wait(self.sample_interval)

    def start(self):
        if not self.available:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.method == "nvml_total_energy":
            self._start_mj = self._read_total_mj()
        else:
            self._samples = []
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        self._t0 = time.time()

    def stop(self):
        if not self.available:
            self.joules = float("nan")
            self.seconds = float("nan")
            return self.joules
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.time() - self._t0

        if self.method == "nvml_total_energy":
            self.joules = (self._read_total_mj() - self._start_mj) / 1000.0
        else:
            self._stop.set()
            self._thread.join(timeout=2.0)
            # trapezoidal integration of power over time
            total = 0.0
            for (t0, p0), (t1, p1) in zip(self._samples, self._samples[1:]):
                total += 0.5 * (p0 + p1) * (t1 - t0)
            self.joules = total
        return self.joules


@contextmanager
def measure_energy(device_index=0):
    """`with measure_energy() as m: ...` then read `m.joules` / `m.seconds`."""
    meter = GPUEnergyMeter(device_index)
    meter.start()
    try:
        yield meter
    finally:
        meter.stop()


def joules_to_kwh(joules):
    return joules / 3.6e6


def carbon_grams(joules, grid_intensity=DEFAULT_GRID_INTENSITY):
    """gCO2e for a given GPU energy draw.

    Counts GPU energy only -- not host CPU, RAM, cooling or PUE. State that
    scoping in the paper; it makes the number a lower bound rather than a
    wrong one.
    """
    return joules_to_kwh(joules) * grid_intensity


# --------------------------------------------------------------------------
# flops
# --------------------------------------------------------------------------

def count_flops(model, sample_input, label=None):
    """Static forward FLOPs for one batch.

    Returns (flops, reliable). `reliable` is False when the counter almost
    certainly under-counted -- most importantly for nn.LSTM, which dispatches
    to a single fused cuDNN/oneDNN kernel that the dispatcher-level counter
    cannot see through. Recurrent baselines therefore get an analytic count
    instead; everything else is measured.
    """
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    counter = FlopCounterMode(display=False)
    try:
        with counter:
            with torch.no_grad():
                model(sample_input, label)
        flops = counter.get_total_flops()
    except Exception as exc:
        warnings.warn(f"FLOP counting failed: {exc}")
        return float("nan"), False

    has_rnn = any(isinstance(m, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN))
                  for m in model.modules())
    if has_rnn:
        analytic = _analytic_rnn_flops(model, sample_input)
        if analytic > 0:
            return flops + analytic, True
        return flops, False

    # A count of zero means the tracer saw nothing at all.
    return flops, flops > 0


def _analytic_rnn_flops(model, sample_input):
    """Closed-form FLOPs for RNN layers the tracer misses.

    An LSTM layer costs 4 gate matmuls on the input (input_size x hidden) and
    4 on the recurrent state (hidden x hidden), per timestep, per sequence.
    Multiply-accumulate counts as 2 FLOPs.
    """
    total = 0
    batch = sample_input.shape[0]
    nodes = sample_input.shape[2]
    steps = sample_input.shape[1]
    # LargeST models fold the node axis into the batch before the recurrence
    sequences = batch * nodes

    gate_mult = {torch.nn.LSTM: 4, torch.nn.GRU: 3, torch.nn.RNN: 1}
    for module in model.modules():
        for cls, gates in gate_mult.items():
            if isinstance(module, cls):
                h = module.hidden_size
                for layer in range(module.num_layers):
                    in_size = module.input_size if layer == 0 else h
                    per_step = 2 * gates * (in_size * h + h * h)
                    total += per_step * steps * sequences
                break
    return total


# --------------------------------------------------------------------------
# latency
# --------------------------------------------------------------------------

def measure_latency(model, sample_input, label=None, warmup=10, iters=50):
    """Median forward-pass latency in ms, timed with CUDA events.

    Median rather than mean because Colab hosts are shared and the tail is
    contaminated by other tenants.
    """
    model.eval()
    device = next(model.parameters()).device
    if device.type != "cuda":
        times = []
        with torch.no_grad():
            for _ in range(warmup):
                model(sample_input, label)
            for _ in range(iters):
                t0 = time.perf_counter()
                model(sample_input, label)
                times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        return times[len(times) // 2]

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(sample_input, label)
        torch.cuda.synchronize()
        for _ in range(iters):
            starter.record()
            model(sample_input, label)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))
    times.sort()
    return times[len(times) // 2]


# --------------------------------------------------------------------------
# result record
# --------------------------------------------------------------------------

@dataclass
class RunRecord:
    """One (model, dataset, seed) run. Serialised to results/*.json."""
    model: str = ""
    dataset: str = ""
    seed: int = 0
    params: int = 0

    # accuracy, averaged over the 12-step horizon
    mae: float = float("nan")
    rmse: float = float("nan")
    mape: float = float("nan")
    horizon_mae: list = field(default_factory=list)
    horizon_rmse: list = field(default_factory=list)
    horizon_mape: list = field(default_factory=list)

    # cost
    train_joules: float = float("nan")
    train_seconds: float = float("nan")
    epochs_run: int = 0
    epoch_seconds: float = float("nan")
    infer_joules_per_1k: float = float("nan")
    latency_ms: float = float("nan")
    flops_per_sample: float = float("nan")
    flops_reliable: bool = False

    energy_method: str = ""
    gpu_name: str = ""
    notes: str = ""

    @property
    def train_kwh(self):
        return joules_to_kwh(self.train_joules)

    def carbon_g(self, grid_intensity=DEFAULT_GRID_INTENSITY):
        return carbon_grams(self.train_joules, grid_intensity)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as fh:
            return RunRecord(**json.load(fh))


def gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"
