#!/usr/bin/env python
"""
h3_benchmarks.py
----------------
Benchmarks and diagnostics for H3, computed with the data path of one sample (the sample folder
given as argument) from the saved test-period fits of the main experiments, and printed as
JSON for make_tables.py. One row per test asset and network.

ridge       ridge regression on the day-t inputs (standardized, intercept not penalized). The penalty
            is chosen on the validation block from a fit on the training block, as the networks'
            hyperparameters are, and the model is refit on the estimation block (training and
            validation): test R2 and Diebold-Mariano statistics against the linear model and the
            selected dropout model.
roll        linear regression re-estimated every test day on the previous 250 (500) trading days:
            test R2 and Diebold-Mariano statistic against the selected dropout model. Unlike every
            other model, it uses returns of the test block up to the day before.
roll_exp    linear regression re-estimated every 63 trading days (a quarter) on all past days, the
            way networks are usually re-trained: test R2.
short_frozen linear regression estimated once on the last 250 trading days before the test block:
            test R2 (a short memory without updating).
lin         how linear a network's test-block fit is in the day-t inputs: the R2 of the fit regressed on
            them (lin_none, lin_sel, lin_wd, lin_es; nan for a constant fit).
chow        Wald test that the intercept and the loadings of the linear regression are the same in the
            estimation and the test block: regression on both blocks with a test-block dummy and its
            interactions with the inputs, Newey-West covariance (lag floor(4 (n/100)^(2/9))).
track       how far the selected dropout model follows the shift in the loadings. b_N are the OLS
            loadings of its test-period fit on the test inputs. With the centered components
            m = X(b_N - b_E) and s = X(b_T - b_E), track = <m, s> / <s, s>: 0 for a model with the
            estimation-block loadings b_E, 1 for one with the test-block loadings b_T. m is split
            into a proportional shrinkage of the linear model's explained component X b_E and the
            rest; track_rest is the share of the shift reproduced by the rest. track_cond: the
            same for the conditional linear model of loading_instability.py.
Diebold-Mariano statistics (src/evaluation.dm_test): positive values favor the model named first
(dm_sel_ridge: network over ridge; dm_ridge_ols: ridge over linear; dm_roll_sel: rolling over network).

    python 3_analysis/h3_benchmarks.py 1_commodity_futures
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

root = sys.argv[1]
sys.path.insert(0, root)
from src.config import EXPERIMENTS  # noqa: E402
from src.data import build_dataset, prepare_splits  # noqa: E402
from src.evaluation import dm_test  # noqa: E402

PERIOD = '2010_2025'
RIDGE_GRID = np.logspace(-4, 1, 26)      # penalty per observation, on standardized inputs
WINDOWS = (250, 500)
QUARTER = 63


def ols(X, y):
    Z = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(Z, y, rcond=None)[0]


def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def ridge(X, y, lam, X_new):
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    b = np.linalg.solve(Z.T @ Z + lam * len(y) * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return y.mean() + ((X_new - mu) / sd) @ b


def chow(X, y, est, test):
    idx = np.concatenate([est, test])
    D = np.r_[np.zeros(len(est)), np.ones(len(test))]
    Z = np.column_stack([np.ones(len(idx)), X[idx], D, D[:, None] * X[idx]])
    b = np.linalg.lstsq(Z, y[idx], rcond=None)[0]
    U = Z * (y[idx] - Z @ b)[:, None]
    n = len(idx)
    lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    S = U.T @ U / n
    for k in range(1, lags + 1):
        G = U[k:].T @ U[:-k] / n
        S += (1 - k / (lags + 1)) * (G + G.T)
    Q = np.linalg.inv(Z.T @ Z / n)
    V = Q @ S @ Q / n
    q = X.shape[1] + 1                                   # dummy and its interactions
    w = float(b[-q:] @ np.linalg.solve(V[-q:, -q:], b[-q:]))
    return w, float(chi2.sf(w, q))


def linear_share(X_T, fit):
    if np.var(fit) < 1e-12:
        return float('nan')
    return r2(fit, np.column_stack([np.ones(len(fit)), X_T]) @ ols(X_T, fit))


def track(X_T, fit, b_E, b_T):
    b_N = ols(X_T, fit)
    m = X_T @ (b_N - b_E)[1:]
    s = X_T @ (b_T - b_E)[1:]
    base = X_T @ b_E[1:]
    m, s, base = m - m.mean(), s - s.mean(), base - base.mean()
    rest = m - (m @ base / (base @ base)) * base
    return float(m @ s / (s @ s)), float(rest @ s / (s @ s))


rows = []
for exp in sorted(EXPERIMENTS):
    if not exp.endswith(PERIOD):
        continue
    try:
        df, X = build_dataset(exp)
    except Exception:  # noqa: BLE001  (an experiment whose data are not available)
        continue
    asset = EXPERIMENTS[exp]['asset']
    Xt, y = X[:, -1, :], df['target'].values
    d = prepare_splits(df, X)
    tr, va, te = d['train']['idx'], d['val']['idx'], d['test']['idx']
    est = np.concatenate([tr, va])
    y_T, X_T = y[te], Xt[te]
    b_E, b_T = ols(Xt[est], y[est]), ols(X_T, y_T)
    lam = RIDGE_GRID[int(np.argmax([r2(y[va], ridge(Xt[tr], y[tr], l, Xt[va])) for l in RIDGE_GRID]))]
    fit_ridge = ridge(Xt[est], y[est], lam, X_T)
    fit_roll = {W: np.array([np.r_[1.0, Xt[t]] @ ols(Xt[t - W:t], y[t - W:t]) for t in te]) for W in WINDOWS}
    n0 = te[0]
    fit_short = np.column_stack([np.ones(len(te)), X_T]) @ ols(Xt[n0 - WINDOWS[0]:n0], y[n0 - WINDOWS[0]:n0])
    fit_exp = np.concatenate([np.column_stack([np.ones(len(te[j:j + QUARTER])), Xt[te[j:j + QUARTER]]])
                              @ ols(Xt[:te[j]], y[:te[j]]) for j in range(0, len(te), QUARTER)])
    vol = pd.Series(y).rolling(20).std().shift(1).values
    s_ = (vol - np.nanmean(vol[est])) / np.nanstd(vol[est])
    est_ok = est[~np.isnan(s_[est])]
    Xc = np.column_stack([Xt, Xt * s_[:, None], s_])
    fit_cond = np.column_stack([np.ones(len(te)), Xc[te]]) @ ols(Xc[est_ok], y[est_ok])
    chow_w, chow_p = chow(Xt, y, est, te)
    for model in ['lstm', 'gru', 'mlp']:
        rd = os.path.join(root, 'results', model, exp)
        if not os.path.isfile(os.path.join(rd, 'run_info.json')):
            continue
        pred = json.load(open(os.path.join(rd, 'predictions.json')))
        assert pred['meta']['dates_test'] == list(d['test']['dates']) and np.allclose(pred['meta']['y_test'], y_T), exp
        fit_ols = np.asarray(pred['benchmarks']['ols_factors'], float)
        assert np.allclose(fit_ols, np.column_stack([np.ones(len(te)), X_T]) @ b_E, atol=1e-6), exp
        sel = json.load(open(os.path.join(rd, 'selection.json')))['best_dropout']
        fit_sel = np.asarray(pred['models'][f"{sel['Dropout_Type']}__{sel['Divergence']}"]['test'], float)
        e_ols, e_sel, e_ridge = y_T - fit_ols, y_T - fit_sel, y_T - fit_ridge
        fits = {'none': np.asarray(pred['models']['none__none']['test'], float), 'sel': fit_sel}
        ad = os.path.join(root, 'results_altreg', model, exp)
        if os.path.isfile(os.path.join(ad, 'predictions.json')):
            alt = json.load(open(os.path.join(ad, 'predictions.json')))['models']
            fits.update(wd=np.asarray(alt['weight_decay__none']['test'], float),
                        es=np.asarray(alt['early_stopping__none']['test'], float))
        tr_sel, tr_sel_rest = track(X_T, fit_sel, b_E, b_T)
        tr_cond, tr_cond_rest = track(X_T, fit_cond, b_E, b_T)
        row = dict(asset=asset, model=model, ridge_lambda=float(lam), ridge_test=r2(y_T, fit_ridge),
                   dm_sel_ridge=dm_test(e_ridge, e_sel)['DM_stat'], dm_ridge_ols=dm_test(e_ols, e_ridge)['DM_stat'],
                   chow=chow_w, chow_p=chow_p, track=tr_sel, track_rest=tr_sel_rest,
                   track_cond=tr_cond, track_cond_rest=tr_cond_rest, roll_exp_test=r2(y_T, fit_exp),
                   short_frozen_test=r2(y_T, fit_short), **{f'lin_{k}': linear_share(X_T, f) for k, f in fits.items()})
        for W in WINDOWS:
            row[f'roll{W}_test'] = r2(y_T, fit_roll[W])
            row[f'dm_roll{W}_sel'] = dm_test(e_sel, y_T - fit_roll[W])['DM_stat']
        rows.append(row)
print(json.dumps(rows))
