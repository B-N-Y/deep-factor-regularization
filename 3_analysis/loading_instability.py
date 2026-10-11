#!/usr/bin/env python
"""
loading_instability.py
----------------------
Diagnostics for H3, computed with the data path of one sample (the sample folder given as
argument) and printed as JSON for make_tables.py:

instability   how much the explained component of test returns changes when the linear
              factor loadings move from the estimation block to the test block:
                  sqrt(mean_t [x_t'(b_test - b_est)]^2) / sd(r_test),
              where b_est and b_test are OLS loadings (with intercept) on the day-t inputs.
              An ex-post diagnostic: it uses the test block and plays no role in any model choice.
cond_gain     test R2 of a conditional linear model minus that of the linear model. In the
              conditional model the loadings vary linearly with the asset's own realized
              volatility over the previous 20 days, standardized on the estimation block:
                  r_t = a + b'x_t + c'(x_t s_{t-1}) + d s_{t-1} + e_t.

ex_ante       the instability measure between the training and the validation block, i.e. with
              information available before the test block.
b_est, b_test the OLS loadings (without intercept) on the estimation and the test block, with the
              input names in `inputs` (main experiments only).
ols_train,    R2 of the linear model estimated on the training block, on the training and the
ols_val       validation block (comparable with the networks' search-stage Train_R2 and Val_R2).
olswin_test   test R2 of a linear regression on the full input window of the recurrent networks
              (window x inputs regressors), estimated on the estimation block.

Computed for every experiment of the study (main sample and subperiods, with the split of
run_grid_search.py) and for every fold of the expanding-window evaluation (run_expanding_cv.py).

    python 3_analysis/loading_instability.py 1_commodity_futures [n_folds]
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

root = sys.argv[1]
n_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
sys.path.insert(0, root)
from src.config import EXPERIMENTS  # noqa: E402
from src.data import build_dataset, input_cols, prepare_splits  # noqa: E402


def ols(X, y):
    Z = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return b


def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def diagnostics(Xt, y, est, test):
    b_est = ols(Xt[est], y[est])
    b_test = ols(Xt[test], y[test])
    Zt = np.column_stack([np.ones(len(test)), Xt[test]])
    instability = float(np.sqrt(np.mean((Zt @ (b_test - b_est)) ** 2)) / y[test].std())
    vol = pd.Series(y).rolling(20).std().shift(1).values
    s = (vol - np.nanmean(vol[est])) / np.nanstd(vol[est])
    est_ok = est[~np.isnan(s[est])]
    Xc = np.column_stack([Xt, Xt * s[:, None], s])
    b_c = ols(Xc[est_ok], y[est_ok])
    yhat_c = np.column_stack([np.ones(len(test)), Xc[test]]) @ b_c
    return instability, r2(y[test], yhat_c) - r2(y[test], Zt @ b_est)


rows = []
for exp in sorted(EXPERIMENTS):
    try:
        df, X = build_dataset(exp)
    except Exception:  # noqa: BLE001  (an experiment whose data are not available)
        continue
    Xt, y = X[:, -1, :], df['target'].values
    d = prepare_splits(df, X)
    est = np.concatenate([d['train']['idx'], d['val']['idx']])
    inst, cg = diagnostics(Xt, y, est, d['test']['idx'])
    ex_ante, _ = diagnostics(Xt, y, d['train']['idx'], d['val']['idx'])   # before the test block
    asset = EXPERIMENTS[exp]['asset']
    period = exp[len(asset) + 1:]
    te, tr, va = d['test']['idx'], d['train']['idx'], d['val']['idx']
    Xw = X.reshape(len(X), -1)
    fit = lambda A, idx_fit, idx_new: np.column_stack([np.ones(len(idx_new)), A[idx_new]]) @ ols(A[idx_fit], y[idx_fit])
    rows.append(dict(asset=asset, period=period, fold=0, instability=inst, cond_gain=cg, ex_ante=ex_ante,
                     ols_train=r2(y[tr], fit(Xt, tr, tr)), ols_val=r2(y[va], fit(Xt, tr, va)),
                     olswin_test=r2(y[te], fit(Xw, est, te)),
                     inputs=input_cols(asset), b_est=ols(Xt[est], y[est])[1:].tolist(),
                     b_test=ols(Xt[te], y[te])[1:].tolist()))
    if period == '2010_2025':
        for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_folds).split(np.arange(len(df))), start=1):
            inst, cg = diagnostics(Xt, y, tr, te)
            rows.append(dict(asset=asset, period=period, fold=k, instability=inst, cond_gain=cg))
print(json.dumps(rows))
