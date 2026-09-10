"""Emit upload-ready notebook variants, one per Kaggle commit.

The full seven-model sweep does not fit inside Kaggle's ~9 h batch cap, so the
run is split across commits: one for the four cheap models, then one per heavy
model. Doing that split by hand means editing two cells in the Kaggle editor
before every commit, and getting it wrong costs a whole run -- leave the
section-6 sweep enabled in a heavy-model commit and you re-train the cheap four
for nothing, or forget to clear TODO and the cheap commit tows STGCN behind it
and blows the cap.

So generate the variants instead. Each output is self-contained: upload, set
Accelerator = GPU T4, attach liuxu77/largest, Save Version > Save & Run All.

Run `python build_notebook.py` first -- this reads its output.
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, 'notebooks', 'ICAISD_LargeST_GreenBench.ipynb')
OUTDIR = os.path.join(HERE, 'notebooks', 'runs')

# name -> (models for the section-6 sweep, TODO list for section 6b)
VARIANTS = {
    'cheap4': ('CHEAP if QUICK_SWEEP else FULL', []),
    'stgcn': ('[]', ['stgcn']),
    'gwnet': ('[]', ['gwnet']),
    'sttn': ('[]', ['sttn']),
    # The patience-15 repair run. hl/nlinear/stid must be re-run under
    # patience 30, but lstm must NOT be: it hit the 100-epoch cap, so patience
    # never fired and its record is already protocol-valid. Routing these three
    # through TODO (not the section-6 sweep) is what picks up cell 6b's
    # PATIENCE = 30 -- and running 'cheap4' instead would retrain lstm for
    # ~2.9 h of GPU to reproduce a record we already have.
    'rerun3': ('[]', ['hl', 'nlinear', 'stid']),
}

SWEEP_RE = re.compile(r'^models = .*$', re.M)
TODO_RE = re.compile(r"^TODO = \[[^\]]*\].*$", re.M)


def _patch(nb, sweep_expr, todo):
    hits = {'sweep': 0, 'todo': 0}
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])

        if 'QUICK_SWEEP' in src and SWEEP_RE.search(src):
            src, n = SWEEP_RE.subn(f'models = {sweep_expr}', src)
            hits['sweep'] += n
        if TODO_RE.search(src):
            listed = ', '.join(repr(m) for m in todo)
            note = ('      # nothing here -- heavy models get their own commit'
                    if not todo else '')
            src, n = TODO_RE.subn(f'TODO = [{listed}]{note}', src)
            hits['todo'] += n

        cell['source'] = src.splitlines(keepends=True)
    return hits


def main():
    with open(MASTER, encoding='utf-8') as fh:
        master = json.load(fh)

    os.makedirs(OUTDIR, exist_ok=True)
    for name, (sweep_expr, todo) in VARIANTS.items():
        nb = json.loads(json.dumps(master))       # deep copy per variant
        hits = _patch(nb, sweep_expr, todo)

        # A silent no-op here produces a notebook that looks right and runs the
        # wrong models, which is the exact failure this script exists to stop.
        assert hits['sweep'] == 1, f'{name}: patched {hits["sweep"]} sweep lines, expected 1'
        assert hits['todo'] == 1, f'{name}: patched {hits["todo"]} TODO lines, expected 1'

        out = os.path.join(OUTDIR, f'ICAISD_{name}.ipynb')
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump(nb, fh, indent=1)
        print(f'wrote {out}  (sweep={sweep_expr}, TODO={todo})')


if __name__ == '__main__':
    main()
