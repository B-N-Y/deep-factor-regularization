"""
data.py
-------
Single data path for every experiment (contemporaneous factor models of
commodity futures).

build_dataset() returns one row per trading day t of the sample period with
    target      excess return of the test asset on day t (%); futures returns are
                excess returns already (config.ASSETS[...]['rf'] is None)
    lag1        its excess return on day t-1 (%)
    <factors>   factor realizations of day t (%), see config.ASSETS
build_windows() stacks, for every row, the inputs [factors_s, lag1_s] of the last
WINDOW days s = t-WINDOW+1..t (days before the sample start may be used; they are
past inputs). The MLP uses only the last step (s = t).

assert_target_not_in_inputs() perturbs the asset's own return on day t and checks
that no input dated t moves, while the inputs dated t+1 do (through lag1).
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

from .config import ASSETS, EXPERIMENTS, N_SPLITS, WINDOW

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def input_cols(asset: str) -> list:
    return ASSETS[asset]['factors'] + ['lag1']


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(asset: str) -> pd.DataFrame:
    """Date, asset return (%), rf (%) and the asset's factors, sorted by date.

    For futures (spec['rf'] is None) the return is already an excess return and rf is 0.
    """
    spec = ASSETS[asset]
    df = pd.read_csv(os.path.join(DATA_DIR, spec['file']), parse_dates=['Date'])
    df = df.rename(columns={spec['return']: 'ret'})
    df['rf'] = 0.0 if spec.get('rf', 'rf') is None else df[spec.get('rf', 'rf')]
    df = df[['Date', 'ret', 'rf'] + spec['factors']].sort_values('Date')
    return df.drop_duplicates('Date').reset_index(drop=True)


def _full_frame(asset: str, raw: pd.DataFrame) -> pd.DataFrame:
    """All days of the raw file: target, lag1 and factors (before the period filter)."""
    raw = raw.sort_values('Date').reset_index(drop=True)
    out = pd.DataFrame({'Date': raw['Date']})
    out['target'] = raw['ret'] - raw['rf']
    out['lag1'] = out['target'].shift(1)
    for f in ASSETS[asset]['factors']:
        out[f] = raw[f]
    out['ret'] = raw['ret']        # kept only to rebuild price levels from fitted returns; never an input
    out['rf'] = raw['rf']
    return out


def build_dataset(exp_name: str, raw: pd.DataFrame = None, window: int = WINDOW):
    """
    Returns (df, X) for the sample period: df has Date, target and the day-t inputs;
    X has shape (n, window, n_inputs) with the last `window` days of inputs per row.
    """
    exp = EXPERIMENTS[exp_name]
    asset = exp['asset']
    if raw is None:
        raw = load_raw(asset)
    full = _full_frame(asset, raw)
    cols = input_cols(asset)
    vals = full[cols].values
    ok = ~np.isnan(vals).any(axis=1)
    windows = np.full((len(full), window, len(cols)), np.nan)
    for k in range(window):            # windows[t, -1-k] = inputs of day t-k
        windows[k:, window - 1 - k, :] = vals[:len(full) - k]
    valid = ok & ~np.isnan(windows).any(axis=(1, 2)) & full['target'].notna().values
    start, end = exp['period']
    in_period = (full['Date'] >= pd.Timestamp(start)) & (full['Date'] < pd.Timestamp(end))
    keep = (valid & in_period.values)
    df = full[keep].reset_index(drop=True)
    return df, windows[keep]


def assert_target_not_in_inputs(exp_name: str, n_dates: int = 25, seed: int = 0) -> int:
    """
    Perturb the asset's own return on randomly chosen days t. Inputs dated t
    (the whole window) must not move; inputs dated t+1 must move (lag1 = target_t).
    Also checks that a factor perturbation on day t does move the inputs dated t
    (factors are contemporaneous). Raises AssertionError on failure.
    """
    asset = EXPERIMENTS[exp_name]['asset']
    raw = load_raw(asset)
    df0, X0 = build_dataset(exp_name, raw)
    idx = {d: i for i, d in enumerate(df0['Date'])}
    rng = np.random.default_rng(seed)
    dates = rng.choice(df0['Date'].values[:-1], size=min(n_dates, len(df0) - 1), replace=False)
    factor = ASSETS[asset]['factors'][0]
    for d in dates:
        i = idx[pd.Timestamp(d)]
        pert = raw.copy()
        pert.loc[pert['Date'] == d, 'ret'] += 3.7
        df1, X1 = build_dataset(exp_name, pert)
        assert np.allclose(X0[i], X1[i]), f"{exp_name}: own return of {pd.Timestamp(d).date()} enters inputs of the same day"
        assert not np.isclose(df0['target'][i], df1['target'][i]), "perturbation did not reach the target"
        assert not np.allclose(X0[i + 1], X1[i + 1]), f"{exp_name}: lag1 of {df0['Date'][i + 1].date()} ignores day t"

        pert = raw.copy()
        pert.loc[pert['Date'] == d, factor] += 1.3
        _, X2 = build_dataset(exp_name, pert)
        assert not np.allclose(X0[i, -1], X2[i, -1]), f"{exp_name}: factor of day t not used on day t"
    return len(dates)


# ─────────────────────────────────────────────────────────────────────────────
# Splits and scaling
# ─────────────────────────────────────────────────────────────────────────────

def scale_arrays(df: pd.DataFrame, X: np.ndarray, fit_idx: np.ndarray, parts: dict) -> dict:
    """
    Fit RobustScalers on rows fit_idx only (inputs: day-t values of each column;
    target) and transform every part. Returns scaled windows and targets per part,
    plus raw targets, raw day-t inputs and dates.
    """
    n_in = X.shape[2]
    sc_x = RobustScaler().fit(X[fit_idx, -1, :])
    sc_y = RobustScaler().fit(df['target'].values[fit_idx].reshape(-1, 1))
    y = df['target'].values
    out = {'scaler_y': sc_y}
    for name, idx in parts.items():
        Xs = sc_x.transform(X[idx].reshape(-1, n_in)).reshape(len(idx), X.shape[1], n_in)
        out[name] = {
            'idx': idx,
            'X': Xs,
            'y_s': sc_y.transform(y[idx].reshape(-1, 1)).ravel(),
            'y': y[idx],
            'X_raw_t': X[idx, -1, :],
            'dates': df['Date'].dt.strftime('%Y-%m-%d').values[idx],
        }
    return out


def prepare_splits(df: pd.DataFrame, X: np.ndarray, n_splits: int = N_SPLITS) -> dict:
    """Train = blocks 1-5, validation = block 6, test = block 7 of TimeSeriesSplit."""
    splits = list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(len(df))))
    train_idx, val_idx = splits[-2]
    _, test_idx = splits[-1]
    assert train_idx.max() < val_idx.min() and val_idx.max() < test_idx.min()
    return scale_arrays(df, X, train_idx, {'train': train_idx, 'val': val_idx, 'test': test_idx})


def describe_splits(data: dict) -> str:
    lines = []
    for part in ['train', 'val', 'test']:
        d = data[part]['dates']
        lines.append(f"   {part:5s}: {d[0]} -> {d[-1]}  (n={len(d)})")
    return "\n".join(lines)
