# What Does Accuracy Cost?

A compute-aware benchmark of traffic forecasting models on the San Diego subset of
[LargeST](https://github.com/liuxu77/LargeST). Every model trains under an identical budget on
identical hardware while training energy, inference energy, FLOPs, parameter count and inference
latency are **measured rather than estimated**, and reported in the same table as the error
metrics.

This repository is the reproducibility artifact for the paper. It contains the instrumented
harness, the run configurations, and the raw per-run records that every number in the paper is
computed from.

## Why

Traffic forecasting is evaluated almost entirely on error. METR-LA, PEMS-BAY and their successors
standardised the *error* protocol; none of them standardised a *cost* protocol. So the published
record cannot distinguish an architecture that reduces MAE by two percent for free from one that
buys the same two percent with an order of magnitude more energy.

Two findings from this harness illustrate why an estimate is not good enough:

- **Parameter count inverts the true cost ordering.** STGCN has 507,532 trainable parameters
  against Graph WaveNet's 311,164 — 1.63x as many — yet requires **11.7x fewer** FLOPs per
  sample, trains on 3.44x less energy, and answers 1.75x faster. Ranking those two models by
  parameter count gets every cost axis backwards.
- **Cheap models draw less power *while running***, not merely for less time, so their advantage
  compounds beyond their shorter runtime. No analytic FLOP count expresses this.

## Layout

```
greenbench/          the harness -- this is the source of truth
  instrument.py      NVML energy counters, FLOPs, latency, the RunRecord type
  engine.py          instrumented training loop
  runner.py          model registry and per-model configuration
  models_lite.py     HistoricalLast / STID / NLinear
  prepare.py         chunked LargeST-SD subset build (memory-bounded)
  compat.py          patches upstream LargeST for pandas 2/3
  analysis.py        Pareto figures and the paper's LaTeX table
notebooks/           GENERATED -- see below
  runs/              one upload-ready notebook per run configuration
results/             raw per-run JSON records; the paper is built from these
  _stale_patience15/ quarantined runs, see "Protocol integrity" below
figures/             regenerated from results/ by greenbench.analysis
paper/               main.tex, refs.bib, FORMAT_REQUIREMENTS.md
```

`notebooks/*.ipynb` are **generated artifacts**. The `greenbench/` modules ship inside them as
`%%writefile` cells, so the uploaded notebook *is* the code that runs. After touching anything in
`greenbench/`:

```bash
python build_notebook.py     # regenerates the master notebook
python make_run_variants.py  # regenerates notebooks/runs/*
```

Skip this and the notebook silently drifts from the modules.

## Reproducing

The data (~7.8 GB) is not vendored. On Kaggle it mounts read-only, which is why the harness runs
there:

1. Upload a notebook from `notebooks/runs/` — not from anywhere else, see below.
2. Accelerator = **GPU T4**. Not P100: Pascal has no NVML total-energy counter, and without it
   there is no energy measurement at all.
3. Internet ON, and attach the dataset `liuxu77/largest` via **+ Add Input**.
4. **Save Version > Save & Run All (Commit)**. An interactive session is ephemeral —
   `/kaggle/working` dies with the browser tab and the run is lost.

Outputs land under the saved version's Output tab. Each record is also printed to stdout by a
trailing cell, so a run survives as log text even if the zip download misbehaves.

Analysis needs no GPU:

```python
from greenbench import analysis
records = analysis.load_results('results')
print(analysis.latex_table(records))
analysis.plot_pareto(records, 'train_joules', out='figures/pareto_train_joules')
```

## Measurement notes

**Energy** is read from the device total-energy counter through NVML, which accumulates joules in
hardware. Scoping is **GPU only** — host CPU, RAM, storage, networking, cooling and datacentre
overhead are excluded, so every energy and carbon figure is a lower bound rather than an
estimate padded by an assumed PUE.

**Carbon** is derived, not measured: energy x 195.04 gCO2e/kWh, from EPA
[eGRID2023 Rev 2](https://www.epa.gov/egrid) subregion CAMX (WECC California), total output rate
430.0 lb CO2e/MWh. California, because LargeST is California data. Read the carbon column as a
scaled version of the energy column.

**Inference energy had a bug worth knowing about.** The first implementation measured over a
fixed 30 batches — a few milliseconds of GPU work for the cheapest models, far below the counter's
resolution — and returned exactly **0.000 J**. A zero inference cost is not a defensible claim,
and it distorted the deployment-cost comparison in favour of precisely the models the paper is
inclined to favour. The harness now replays batches until a 5-second wall-clock floor with a
device synchronisation per pass. Records produced before that fix are quarantined, not deleted.

**Training budget** is a maximum of 100 epochs with early stopping at patience 30, identical
across models. A model that reaches the cap never converged, so its energy is a *lower bound* and
its error pessimistic; those rows are daggered in the results table rather than mixed silently
with converged runs.

**Not bit-reproducible.** `runner.py` leaves `cudnn.deterministic = False`, because forcing
deterministic kernels changes the performance characteristics being measured. Measured
run-to-run spread is approximately +/-0.3 MAE. Differences smaller than that are treated as
unresolved rather than as rankings.

## Protocol integrity

`results/_stale_patience15/` holds three records produced under patience 15 and the broken
inference meter. They are kept rather than deleted because the failure is instructive: the
notebook that produced them was a stale copy, so the sweep cell ran at patience 15 while a later
cell ran at patience 30, and the two protocols would have been mixed into one table under
identical filenames without anything raising. `analysis.load_results` does not descend into that
directory.

The rule that came out of it: **upload from `notebooks/runs/`, never from a downloads folder.**

Records marked `"notes": "RECONSTRUCTED from Kaggle notebook log..."` were rebuilt by
`rebuild_records_from_logs.py` from an executed notebook rather than recovered as original JSON.
The reconstruction was validated by checking that `analysis` over the rebuilt records reproduces
the Kaggle-generated LaTeX table digit for digit.

## Status

`results/` currently holds four of the six models in the paper — `hl`, `lstm`, `stgcn`, `gwnet`.
`nlinear` and `stid` are being re-run under the corrected protocol
(`notebooks/runs/ICAISD_rerun3.ipynb`). STTN was dropped: at its measured per-epoch cost it does
not fit the 12-hour compute limit, and a run killed at the limit writes no record.

## License

Harness code is MIT. LargeST is redistributed by its authors under their own terms; the ACM
template pack is excluded from this repository and should be obtained from the conference.
