"""Builds the LargeST-SD subset without blowing up Colab's RAM.

The upstream recipe is: download the whole Kaggle archive, load the full
California frame, slice out District 11. That frame is 8,600 sensors x 105,120
five-minute steps; in float64 it is roughly 7 GB, which a free Colab instance
does not have. Since we only ever want 716 of those columns, this module reads
the file in row blocks and keeps just the columns it needs -- peak memory is a
few hundred MB.

The output is byte-identical in intent to running data/ca/process_ca_his.ipynb
followed by data/sd/generate_sd_dataset.ipynb: District 11 sensors, resampled
to 15 minutes, NaNs zero-filled.
"""

import os

import numpy as np
import pandas as pd


def _decode_index(axis, raw):
    """Rebuild a DatetimeIndex from a raw HDF5 axis, honouring its stored unit.

    The integers under axis1 are counts since the epoch, but the unit is a
    property of the pandas that wrote the file: 1.x always wrote
    datetime64[ns], 3.x writes datetime64[us]. LargeST was published from
    pandas 1.x, so a nanosecond assumption reads the archive correctly and
    then quietly fails on anything regenerated locally -- a 2019 index comes
    back as January 1970, and because it is still monotonic nothing raises.
    The resample simply collapses 2,880 rows into 2. So take the unit the
    file declares rather than assuming one.
    """
    kind = axis.attrs.get('kind', b'datetime64[ns]')
    if isinstance(kind, bytes):
        kind = kind.decode()
    kind = str(kind)
    if kind.startswith('datetime64'):
        # pandas 1.x -- which wrote the published LargeST archive -- stores a
        # bare 'datetime64' with no unit, and numpy will not build a dtype from
        # that. It always meant nanoseconds, so supply the unit it omitted.
        if '[' not in kind:
            kind = 'datetime64[ns]'
        return pd.DatetimeIndex(raw.view(np.dtype(kind)))
    return pd.to_datetime(raw)


def read_hdf_columns(path, wanted_columns, chunk_rows=5000, verbose=True):
    """Read a subset of columns from a pandas HDF5 file, block by block.

    Tries the cheap path first (`pd.read_hdf(columns=...)`, which works for
    table-format files). Falls back to reading the raw h5py datasets, which is
    what fixed-format files require -- they support no partial reads at all
    through pandas.
    """
    try:
        df = pd.read_hdf(path, columns=list(wanted_columns))
        if verbose:
            print(f'read {path} via pandas column selection: {df.shape}')
        return df
    except (TypeError, ValueError, NotImplementedError) as exc:
        if verbose:
            print(f'pandas column selection unavailable ({exc.__class__.__name__}), '
                  f'falling back to chunked h5py read')

    import h5py

    with h5py.File(path, 'r') as fh:
        key = next(iter(fh.keys()))
        grp = fh[key]

        def _decode(arr):
            return [v.decode() if isinstance(v, bytes) else str(v) for v in arr]

        # Fixed-format frames store values per dtype block; the column order
        # inside a block follows block*_items, not axis0.
        blocks = sorted(k for k in grp.keys() if k.endswith('_values'))
        if not blocks:
            raise RuntimeError(f'{path}: no *_values datasets under /{key}')

        axis1 = grp['axis1']
        index_raw = axis1[:]
        n_rows = len(index_raw)

        wanted = [str(c) for c in wanted_columns]
        wanted_set = set(wanted)
        collected = {}

        for values_key in blocks:
            items_key = values_key.replace('_values', '_items')
            if items_key not in grp:
                continue
            names = _decode(grp[items_key][:])
            take = [(i, n) for i, n in enumerate(names) if n in wanted_set]
            if not take:
                continue

            positions = [i for i, _ in take]
            labels = [n for _, n in take]
            dset = grp[values_key]

            if dset.shape[0] != n_rows:
                raise RuntimeError(
                    f'{path}: {values_key} is {dset.shape}, expected {n_rows} rows first; '
                    'this pandas version stores blocks transposed')

            out = np.empty((n_rows, len(positions)), dtype=np.float32)
            for start in range(0, n_rows, chunk_rows):
                stop = min(start + chunk_rows, n_rows)
                out[start:stop] = dset[start:stop, :][:, positions]
            for j, label in enumerate(labels):
                collected[label] = out[:, j]
            if verbose:
                print(f'  {values_key}: pulled {len(positions)} columns')

        missing = wanted_set - set(collected)
        if missing:
            raise KeyError(f'{len(missing)} requested sensors not in {path}, '
                           f'e.g. {sorted(missing)[:5]}')

        df = pd.DataFrame({c: collected[c] for c in wanted},
                          index=_decode_index(axis1, index_raw))

    if verbose:
        print(f'read {path} via chunked h5py: {df.shape}')
    return df


def build_sd_subset(ca_dir, sd_dir, year='2019', resample='15min', verbose=True):
    """Produce sd_meta.csv, sd_rn_adj.npy and sd_his_<year>.h5.

    Mirrors the upstream notebooks. Note the resample default: the reference
    experiments use 15-minute bins (STGODE's --tpd default of 96 confirms it),
    so a day is 96 steps. Pass resample=None to keep the raw 5-minute feed,
    but then every steps_per_day setting downstream must change to 288.
    """
    os.makedirs(sd_dir, exist_ok=True)

    ca_meta = pd.read_csv(os.path.join(ca_dir, 'ca_meta.csv'))
    sd_meta = ca_meta[ca_meta.District == 11].reset_index(drop=True)
    sd_meta.to_csv(os.path.join(sd_dir, 'sd_meta.csv'), index=False)
    if verbose:
        print(f'District 11 sensors: {len(sd_meta)}')

    # adjacency: index into the CA matrix by ID2 (the row position in ca_meta)
    id2 = sd_meta.ID2.values.tolist()
    ca_adj = np.load(os.path.join(ca_dir, 'ca_rn_adj.npy'))
    sd_adj = ca_adj[id2][:, id2]
    np.save(os.path.join(sd_dir, 'sd_rn_adj.npy'), sd_adj)
    if verbose:
        print(f'adjacency {ca_adj.shape} -> {sd_adj.shape}')

    sensor_ids = sd_meta.ID.astype(str).values.tolist()
    raw = os.path.join(ca_dir, f'ca_his_raw_{year}.h5')
    processed = os.path.join(ca_dir, f'ca_his_{year}.h5')
    source = raw if os.path.exists(raw) else processed
    if not os.path.exists(source):
        raise FileNotFoundError(
            f'need {raw} or {processed}; download ca_his_raw_{year}.h5 from Kaggle first')

    sd_his = read_hdf_columns(source, sensor_ids, verbose=verbose)

    if resample and source == raw:
        sd_his = sd_his.resample(resample).mean().round(0)
        if verbose:
            print(f'resampled to {resample}: {sd_his.shape}')

    sd_his = sd_his.fillna(0)
    assert sd_his.isnull().any().sum() == 0, 'nulls survived fillna'

    out = os.path.join(sd_dir, f'sd_his_{year}.h5')
    sd_his.to_hdf(out, key='t', mode='w')
    if verbose:
        print(f'wrote {out}: {sd_his.shape}')
    return sd_his
