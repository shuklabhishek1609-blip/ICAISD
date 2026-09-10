"""Patches LargeST for a modern Python stack.

The repo targets PyTorch 1.12 / pandas 1.x (2023). Colab in 2026 ships pandas
2.x and numpy 2.x, where a couple of the repo's calls are hard errors rather
than warnings. Call `apply_all(repo_root)` once after cloning, before importing
anything from `src`.

Each patch is idempotent and logs what it touched, so a run that silently
skipped a fix is visible rather than mysterious.
"""

import os
import re


def _sub_in_file(path, pattern, replacement, label):
    if not os.path.exists(path):
        return f'SKIP {label}: {path} missing'
    with open(path, encoding='utf-8') as fh:
        src = fh.read()
    new, n = re.subn(pattern, replacement, src)
    if n == 0:
        return f'OK   {label}: nothing to change (already patched?)'
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(new)
    return f'FIX  {label}: {n} replacement(s) in {os.path.basename(path)}'


def patch_pandas_append(repo_root):
    """DataFrame.append was removed in pandas 2.0; use pd.concat.

    Hits generate_data_for_training.py, which builds the multi-year frame with
    `df = df.append(df_tmp)`. Without this the data-generation step dies with
    AttributeError before producing his.npz.
    """
    path = os.path.join(repo_root, 'data', 'generate_data_for_training.py')
    return _sub_in_file(
        path,
        r'df = df\.append\(df_tmp\)',
        'df = pd.concat([df, df_tmp], axis=0) if len(df) else df_tmp',
        'pandas append -> concat')


def patch_np_asarray_adj(repo_root):
    """normalize_adj_mx returns np.matrix via .todense().

    torch.tensor() on an np.matrix keeps the 2-D matrix semantics and newer
    numpy is stricter about it. Returning ndarray avoids a class of shape
    surprises downstream.
    """
    path = os.path.join(repo_root, 'src', 'utils', 'graph_algo.py')
    return _sub_in_file(
        path,
        r'adj = \[a\.astype\(np\.float32\)\.todense\(\) for a in adj\]',
        'adj = [np.asarray(a.astype(np.float32).todense()) for a in adj]',
        'adj todense -> ndarray')


def apply_all(repo_root, verbose=True):
    results = [
        patch_pandas_append(repo_root),
        patch_np_asarray_adj(repo_root),
    ]
    if verbose:
        for line in results:
            print(line)
    return results
