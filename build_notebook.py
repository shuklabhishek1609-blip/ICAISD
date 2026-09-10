"""Generates the Colab notebook from the greenbench modules.

The notebook has to be self-contained -- one file you upload to Colab, no repo
to clone, no Drive mount -- but the modules also have to be editable and
diffable locally. So the modules are the source of truth and the notebook is
built from them: run this after touching anything in greenbench/.

    python build_notebook.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ['__init__', 'instrument', 'compat', 'prepare', 'models_lite',
           'engine', 'runner', 'analysis']
OUT = os.path.join(HERE, 'notebooks', 'ICAISD_LargeST_GreenBench.ipynb')


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip().split('\n')}


def code(text):
    lines = text.strip('\n').split('\n')
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': [l + '\n' for l in lines[:-1]] + [lines[-1]]}


def module_cells():
    """One %%writefile cell per module, contents read from disk."""
    cells = [md("""
## 3. Install `greenbench`

These cells write out the measurement package. They are generated from the
local `greenbench/` sources by `build_notebook.py` -- edit the `.py` files and
regenerate rather than editing here, or the two will drift.
""")]
    for name in MODULES:
        path = os.path.join(HERE, 'greenbench', f'{name}.py')
        with open(path, encoding='utf-8') as fh:
            src = fh.read()
        cells.append(code(f'%%writefile greenbench/{name}.py\n{src}'))
    return cells


NOTEBOOK_HEAD = [
    md("""
# Compute-Aware Traffic Forecasting on LargeST-SD

Companion notebook for the ICAISD 2026 submission (Track 1: Sustainable
Transportation and Smart Cities).

**What this measures.** Every traffic-forecasting paper reports MAE. Almost
none report what the MAE cost. This notebook trains a spread of models on the
same data under the same protocol and records, for each one: accuracy, real
GPU energy draw during training, inference energy, latency, parameters and
FLOPs. The output is a Pareto frontier over accuracy and cost.

**Dataset.** [LargeST](https://github.com/liuxu77/LargeST) (Liu et al., NeurIPS
2023 Datasets & Benchmarks) -- CalTrans PeMS loop detectors. We use the San
Diego subset: 716 sensors, calendar 2019, resampled to 15-minute bins, 12 steps
in / 12 steps out.

**Where to run it.** Works on either Kaggle or Colab; the next cell detects
which and sets its paths accordingly.

*Kaggle (preferred)* -- LargeST is a Kaggle dataset, so it mounts read-only at
`/kaggle/input` and there is nothing to download. Add it via **+ Add Input**,
search `liuxu77/largest`. Set **Accelerator = GPU T4**, and turn **Internet on**
(the repo still has to be cloned). Do not pick P100: its NVML energy counter
does not exist on Pascal, so energy falls back to power sampling.

*Colab* -- needs a Kaggle API token and pulls ~7.8 GB over the network first.

**Cost.** The full seven-model sweep is roughly 3-6 hours on a T4; the
lightweight models alone are under 30 minutes. Start with `QUICK_SWEEP = True`.
"""),

    md('## 1. Environment'),

    code("""
import subprocess, sys

# nvidia-smi is absent on a CPU runtime, and the raw FileNotFoundError that
# raises says nothing about the actual problem. Catch it so the assert below
# gets to deliver the message that tells you what to do.
try:
    print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,power.max_limit',
                          '--format=csv'], capture_output=True, text=True).stdout)
except FileNotFoundError:
    print('nvidia-smi not found -- no GPU is attached to this runtime.')

import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
assert torch.cuda.is_available(), (
    'No GPU attached. Runtime > Change runtime type > T4 GPU, then Run all again. '
    'Energy is measured through NVML, so a CPU run cannot produce the paper numbers.')
"""),

    code("""
# One place where the two hosts differ, so nothing below has to care again.
import os, sys

IN_KAGGLE = os.path.isdir('/kaggle/input')
HOST = 'kaggle' if IN_KAGGLE else 'colab'

# Kaggle names the mount folder after the dataset slug -- usually. Attach it
# through a different UI path and it lands under another name or an extra
# level of nesting ('datasets/...', say). The marker file is what matters,
# so search for that instead of trusting the name.
def _find_ca_dir():
    for base, dirs, files in os.walk('/kaggle/input'):
        if 'ca_meta.csv' in files:
            return base
        dirs[:] = sorted(dirs)[:20]       # datasets are shallow; stay sane
    return None

if IN_KAGGLE:
    ROOT = '/kaggle/working/LargeST'      # writable; /kaggle/input is not
    CA_DIR = _find_ca_dir()               # the dataset, mounted read-only
    OUT = '/kaggle/working'
else:
    ROOT = '/content/LargeST'
    CA_DIR = f'{ROOT}/data/ca'            # downloaded in section 4
    # Colab recycles runtimes and takes /content with them. A 3-6 hour sweep is
    # long enough that this is a matter of when, not if, so results go to Drive.
    # Mounted here rather than at the sweep so it fails now, not four hours in.
    from google.colab import drive
    drive.mount('/content/drive')
    OUT = '/content/drive/MyDrive/icaisd'

RESULTS, FIGURES = f'{OUT}/results', f'{OUT}/figures'

# Logs and checkpoints stay on local disk. The engine writes a checkpoint on
# every validation improvement, and routing that through a Drive mount would
# put network I/O in the training loop. Results are small and written once per
# model, so those are the ones worth persisting.
LOGS = f'{ROOT}/logs'

print(f'host={HOST}\\nroot={ROOT}\\nca_dir={CA_DIR}\\nresults={RESULTS}')

if IN_KAGGLE and CA_DIR is None:
    attached = sorted(os.listdir('/kaggle/input'))
    raise SystemExit(
        f'No folder under /kaggle/input contains ca_meta.csv. '
        f'Attached inputs: {attached or "(none)"}\\n'
        'Add the dataset with "+ Add Input" > search "liuxu77/largest" > Add, '
        'then re-run this cell.')
"""),

    code("""
# tables  -> pandas HDF5 support (LargeST ships .h5)
# nvidia-ml-py -> provides the `pynvml` module used for energy measurement
!pip install --quiet tables nvidia-ml-py h5py

import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
try:
    pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    print('NVML total-energy counter available -- energy measured directly')
except Exception:
    print('NVML total-energy counter missing -- will integrate power samples instead')
"""),

    md("""
## 2. Get LargeST and patch it for a 2026 stack

The repo targets pandas 1.x / PyTorch 1.12. Two calls are hard errors on a 2026
image; `greenbench.compat` fixes them and prints what it changed.

On Kaggle this needs **Internet on** in the sidebar settings, or the clone
fails.
"""),

    code("""
import os
os.makedirs(os.path.dirname(ROOT), exist_ok=True)
os.chdir(os.path.dirname(ROOT))

if not os.path.isdir(ROOT):
    !git clone --quiet --depth 1 https://github.com/liuxu77/LargeST.git {ROOT}

assert os.path.isdir(ROOT), (
    f'clone failed -- {ROOT} does not exist. On Kaggle, turn Internet on in the '
    'sidebar settings (needs a phone-verified account) and re-run.')

# Everything downstream resolves data paths relative to cwd, so stay here.
os.chdir(ROOT)
for d in ['greenbench', RESULTS, FIGURES, LOGS]:
    os.makedirs(d, exist_ok=True)
print('cwd', os.getcwd())
"""),
]


NOTEBOOK_TAIL = [
    code("""
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from greenbench import compat
compat.apply_all(ROOT)
"""),

    md("""
## 4. Get the data

**On Kaggle there is nothing to do here** -- the dataset is already mounted
read-only at `/kaggle/input/largest`. Both cells below no-op and you can move
straight to section 5.

**On Colab** you need an API token: kaggle.com > your profile > Settings > API.
Kaggle issues two formats depending on account age -- a single bearer token
(`KGAT_...`, saved to `~/.kaggle/access_token`) or the older `kaggle.json` with
a username and key. The cell takes either.

Put the token in Colab's **secret manager**, not in a cell: key icon in the left
sidebar > add `KAGGLE_API_TOKEN` > enable "Notebook access". A token typed into
a cell is saved inside the `.ipynb` and travels to anyone you send it to.

We pull only the three files the SD subset needs -- the 2019 history, the sensor
metadata and the road-network adjacency. That is still ~7.8 GB, most of it the
history file; the full five-year archive is roughly 30 GB.
"""),

    code("""
import os

if IN_KAGGLE:
    print('Kaggle: dataset mounted, no token or download needed.')
    print(sorted(os.listdir(CA_DIR)))
else:
    !pip install --quiet kaggle

    from getpass import getpass
    token = None
    try:
        from google.colab import userdata
        token = userdata.get('KAGGLE_API_TOKEN')
    except Exception:
        pass                  # secret not set, or not running on Colab
    if not token:
        token = getpass('Kaggle API token (KGAT_...): ').strip()

    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    token_path = os.path.expanduser('~/.kaggle/access_token')
    with open(token_path, 'w') as fh:
        fh.write(token.strip())
    os.chmod(token_path, 0o600)

    # Fails loudly here rather than three cells later if the token is wrong.
    !kaggle datasets files liuxu77/largest
"""),

    code("""
# Pull the three files we need. These names were checked against the live
# Kaggle listing on 2026-08-02; if the cell above ever prints something
# different, correct it here rather than downloading the whole archive.
# ca_his_raw_2019.h5 is 7.2 GB, so this cell is the slow one.
import glob, os, zipfile

YEAR = '2019'
NEEDED = [f'ca_his_raw_{YEAR}.h5', 'ca_meta.csv', 'ca_rn_adj.npy']

if not IN_KAGGLE:
    os.makedirs(CA_DIR, exist_ok=True)
    os.chdir(CA_DIR)

    for fname in NEEDED:
        if os.path.exists(fname):
            print(f'have {fname}')
            continue
        !kaggle datasets download liuxu77/largest -f {fname} --force
        # Single-file downloads sometimes arrive zipped. Both globs can match
        # the same archive, so dedupe them -- otherwise the second pass tries
        # to open the zip the first pass already extracted and deleted, and
        # the whole loop dies *after* a 7 GB download has succeeded.
        for z in sorted(set(glob.glob(f'{fname}.zip')) | set(glob.glob('*.zip'))):
            if not os.path.exists(z):
                continue
            with zipfile.ZipFile(z) as zf:
                zf.extractall('.')
            os.remove(z)

    os.chdir(ROOT)

missing = [f for f in NEEDED if not os.path.exists(os.path.join(CA_DIR, f))]
assert not missing, f'missing from {CA_DIR}: {missing}'
!ls -lh {CA_DIR}
"""),

    md("""
## 5. Build the SD subset

Upstream does this by loading the whole California frame and slicing District
11 out of it -- about 7 GB resident, which a free Colab instance cannot hold.
`greenbench.prepare` reads the file in row blocks and keeps only the 716
columns it needs.
"""),

    code("""
from greenbench.prepare import build_sd_subset

# ca_dir is read-only on Kaggle -- build_sd_subset only reads from it, and
# every write goes to sd_dir under the working tree.
sd = build_sd_subset(ca_dir=CA_DIR, sd_dir=f'{ROOT}/data/sd', year=YEAR,
                     resample='15min')
print(sd.index[0], '->', sd.index[-1])
sd.iloc[:5, :6]
"""),

    code("""
# Window the series into (12 in -> 12 out) samples with 60/20/20 splits.
# Writes data/sd/2019/{his.npz, idx_train.npy, idx_val.npy, idx_test.npy}.
import os
os.chdir(f'{ROOT}/data')
!python generate_data_for_training.py --dataset sd --years {YEAR}
os.chdir(ROOT)

import numpy as np
ptr = np.load(f'data/sd/{YEAR}/his.npz')
print('data', ptr['data'].shape, '| mean', float(ptr['mean']), '| std', float(ptr['std']))
for split in ['train', 'val', 'test']:
    print(split, np.load(f'data/sd/{YEAR}/idx_{split}.npy').shape)
"""),

    md("""
## 6. Run the sweep

One code path builds every model, so batch size, input dimension, loss,
splits and early-stopping rule are identical across the lineup -- the only
thing that varies is the architecture. Per-model hyperparameters are the
upstream defaults; no baseline has been tuned down.

Results are written to `results/*.json` after each run, so a disconnect costs
you one model rather than the sweep.
"""),

    code("""
import os
from greenbench.runner import run_experiment, MODEL_REGISTRY

QUICK_SWEEP = True   # True: cheap models only, ~30 min. False: everything.
SEEDS = [2023]       # add 2024, 2025 before the camera-ready
MAX_EPOCHS = 100

# Upstream LargeST uses 30. Dropping it to 15 cuts off slow-converging models
# -- which is exactly the expensive class this paper argues against -- so the
# protocol would be quietly biased toward its own conclusion. Keep 30 and pay
# the runtime; every model in the table must use the same value, including the
# ones run one at a time in section 6b.
PATIENCE = 30

CHEAP = ['hl', 'nlinear', 'stid', 'lstm']
FULL = CHEAP + ['stgcn', 'gwnet', 'sttn']
models = CHEAP if QUICK_SWEEP else FULL

print('will run:', models)
print('results ->', RESULTS)
""" ),

    code("""
import traceback

records = []
for seed in SEEDS:
    for name in models:
        # Already-finished models are skipped, so re-running this cell after a
        # disconnect resumes the sweep instead of restarting it.
        done = os.path.join(RESULTS, f'{name}_SD_{YEAR}_s{seed}.json')
        if os.path.exists(done):
            print(f'skip {name} seed={seed} -- already in {RESULTS}')
            continue

        print(f'\\n{"=" * 60}\\n{name}  seed={seed}\\n{"=" * 60}')
        try:
            rec = run_experiment(name, dataset='SD', years=YEAR, seed=seed,
                                 bs=64, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                                 results_dir=RESULTS, log_root=LOGS)
            records.append(rec)
            print(f'  MAE {rec.mae:.3f} | train {rec.train_joules / 1000:.1f} kJ '
                  f'({rec.epochs_run} epochs) | {rec.latency_ms:.1f} ms | {rec.params:,} params')
        except Exception:
            # One failed model should not end the sweep.
            traceback.print_exc()
"""),

    md("""
### 6b. Run selected models

The heavy models do not fit in one session, so run them a few at a time with
this cell instead of the one above. Edit `TODO`, run, download the JSONs.

Anything already present in `RESULTS` is skipped, so cancelling and re-running
resumes rather than restarts -- **within a live session**. A session that ends
takes `/kaggle/working` with it, and then nothing is left to skip; that is what
the downloaded JSONs are for.
"""),

    code("""
import os, traceback

# This cell stands alone, so it is the one people jump straight to after a
# restart -- at which point the modules are not on disk yet and the import
# below fails with a bare ModuleNotFoundError that says nothing useful.
missing = [n for n in ('ROOT', 'RESULTS', 'LOGS', 'YEAR') if n not in globals()]
assert not missing, (
    f'{missing} undefined -- sections 1-5 have not run in this session. '
    'Select the section 6 cell and use Run > Run Before, then come back here.')

from greenbench.runner import run_experiment, MODEL_REGISTRY

TODO = ['stgcn']      # e.g. ['hl', 'nlinear', 'stid'] or ['gwnet'] or ['sttn']
SEED = 2023
MAX_EPOCHS = 100

PATIENCE = 30         # same value as section 6 -- see the note there

unknown = [m for m in TODO if m not in MODEL_REGISTRY]
assert not unknown, f'unknown: {unknown}; have {sorted(MODEL_REGISTRY)}'

for name in TODO:
    done = os.path.join(RESULTS, f'{name}_SD_{YEAR}_s{SEED}.json')
    if os.path.exists(done):
        print(f'skip {name} -- already in {RESULTS}')
        continue

    print(f'\\n{"=" * 60}\\n{name}  seed={SEED}  patience={PATIENCE}\\n{"=" * 60}')
    try:
        rec = run_experiment(name, dataset='SD', years=YEAR, seed=SEED,
                             bs=64, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                             results_dir=RESULTS, log_root=LOGS)
        print(f'  MAE {rec.mae:.3f} | train {rec.train_joules / 1000:.1f} kJ '
              f'({rec.epochs_run} ep) | infer/1k {rec.infer_joules_per_1k:.3f} J '
              f'| {rec.latency_ms:.2f} ms | {rec.params:,} params')
    except Exception:
        traceback.print_exc()

print('\\nfiles in', RESULTS)
print(sorted(os.listdir(RESULTS)))
"""),

    md('## 7. Figures and table'),

    code("""
import traceback

from greenbench import analysis

recs = analysis.load_results(RESULTS, dataset='SD')
print(f'{len(recs)} runs loaded:', sorted(r.model for r in recs))


def _try(label, fn, *a, **kw):
    \"\"\"Run a figure/table step without letting it fail the notebook.

    In a committed (Save & Run All) Kaggle run an uncaught exception aborts the
    version, and hours of finished training go with it. Every step below is
    derived output that can be regenerated locally from the JSONs in seconds --
    so nothing here is worth losing a run over. plot_pareto in particular
    raises outright when no record has a finite cost, which is the normal case
    for a single-model commit whose inference meter returned NaN.
    \"\"\"
    try:
        return fn(*a, **kw)
    except Exception:
        print(f'-- {label} failed (results are still saved); traceback follows')
        traceback.print_exc()


_try('pareto/train', analysis.plot_pareto, recs, cost='train_joules',
     out=f'{FIGURES}/pareto_train')
_try('pareto/infer', analysis.plot_pareto, recs, cost='infer_joules_per_1k',
     out=f'{FIGURES}/pareto_infer')
_try('horizon', analysis.plot_horizon, recs, out=f'{FIGURES}/horizon')
"""),

    code("""
table = _try('latex_table', analysis.latex_table, recs)
if table:
    print(table)
    with open(f'{FIGURES}/main_table.tex', 'w') as fh:
        fh.write(table)
"""),

    code("""
# On Kaggle everything under /kaggle/working is already saved with the notebook
# version -- use "Save Version > Save & Run All" and collect it from the Output
# tab. On Colab the results are in Drive; this is a convenience copy.
import os, shutil

# Stage in /tmp: zipping a directory into itself makes the archive walk its own
# growing output.
staging = '/tmp/icaisd_bundle'
shutil.rmtree(staging, ignore_errors=True)
os.makedirs(staging)
for src in (RESULTS, FIGURES):
    if os.path.isdir(src):
        shutil.copytree(src, os.path.join(staging, os.path.basename(src)))

archive = shutil.copy(shutil.make_archive('/tmp/icaisd_results', 'zip', staging), OUT)
print('wrote', archive)

if not IN_KAGGLE:
    from google.colab import files
    files.download(archive)
"""),

    md("""
## Before this becomes a paper

Things that are deliberately unfinished here:

- **Seeds.** One seed is a pilot, not a result. Run at least three and report
  mean +/- std; a 0.05 MAE gap across models means nothing next to seed noise.
- **Carbon intensity.** `instrument.DEFAULT_GRID_INTENSITY` is a placeholder.
  Replace it with a sourced CAISO figure and cite it -- LargeST is California
  data, so a California grid factor is the defensible choice.
- **Energy scope.** We measure GPU draw only: no CPU, no DRAM, no PUE. Say so.
  It makes the numbers a lower bound rather than an overstatement.
- **Shared hardware.** Colab hosts are multi-tenant, so energy readings carry
  noise you do not control. Report the GPU model, run the sweep in one session
  where possible, and treat order-of-magnitude gaps as the finding rather than
  10% differences.
- **Excluded baselines.** STGODE (hours of DTW preprocessing) and the six
  models needing custom engines (DCRNN, AGCRN, ASTGCN, DGCRN, DSTAGNN,
  D2STGNN) are not in the lineup. State that plainly; do not imply the sweep
  is exhaustive.
"""),
]


def main():
    cells = NOTEBOOK_HEAD + module_cells() + NOTEBOOK_TAIL
    nb = {
        'cells': cells,
        'metadata': {
            'accelerator': 'GPU',
            'colab': {'provenance': [], 'toc_visible': True},
            'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
            'language_info': {'name': 'python'},
        },
        'nbformat': 4,
        'nbformat_minor': 0,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(nb, fh, indent=1)
    print(f'wrote {OUT} ({len(cells)} cells)')


if __name__ == '__main__':
    main()
