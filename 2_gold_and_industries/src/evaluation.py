"""
evaluation.py
-------------
Prediction, explanatory-fit metrics, linear factor-model benchmarks and the
Diebold-Mariano comparison.

R2        1 - SSE / sum((y - mean(y))^2)       usual (demeaned) R^2 on the evaluation window
R2_total  1 - SSE / sum(y^2)                    "total R^2" of Gu, Kelly and Xiu (2021)
R2_vs_ols 1 - SSE(model) / SSE(OLS factor model)

Sign convention: dm_test(e1, e2) uses d_t = L(e1_t) - L(e2_t); a positive
statistic means the second model fits better.
"""
import numpy as np
import scipy.stats as sps
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import MC_SAMPLES

# Benchmark used for R2_vs_ols and the headline DM comparison
LINEAR = 'ols_factors'


def predict(model, X: np.ndarray, scaler_y, device: str, mc_samples: int = MC_SAMPLES) -> np.ndarray:
    """Predictions in original units (%). MC dropout averages mc_samples passes."""
    model.eval()
    x = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        if model.dropout_type == 'monte_carlo':
            out = torch.stack([model(x, mc=True)[0] for _ in range(mc_samples)]).mean(0)
        else:
            out, _ = model(x)
    return scaler_y.inverse_transform(out.cpu().numpy().reshape(-1, 1)).ravel()


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    sse = float(np.sum((y - yhat) ** 2))
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y, yhat))),
        'MAE': float(mean_absolute_error(y, yhat)),
        'R2': float(r2_score(y, yhat)),
        'R2_total': float(1.0 - sse / np.sum(y ** 2)),
    }


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2) -> dict:
    """
    Diebold and Mariano (1995) test with the Harvey, Leybourne and Newbold (1997)
    small-sample correction. Long-run variance of d_t uses autocovariances up to
    lag h-1. Two-sided p-value from a t(n-1) distribution.
    """
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    d_mean = d.mean()
    dc = d - d_mean
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(h)]
    lrv = gamma[0] + 2.0 * sum(gamma[1:])
    if lrv <= 0:
        return {'DM_stat': np.nan, 'DM_pvalue': np.nan, 'mean_loss_diff': float(d_mean)}
    stat = d_mean / np.sqrt(lrv / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    pvalue = 2.0 * sps.t.sf(abs(stat), df=n - 1)
    return {'DM_stat': float(stat), 'DM_pvalue': float(pvalue), 'mean_loss_diff': float(d_mean)}


def _ols_fit_predict(X_fit: np.ndarray, y_fit: np.ndarray, X_new: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X_fit)), X_fit])
    beta, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
    return np.column_stack([np.ones(len(X_new)), X_new]) @ beta


def benchmark_fits(X_t: np.ndarray, y: np.ndarray, fit_idx: np.ndarray, eval_idx: np.ndarray) -> dict:
    """
    Linear benchmarks estimated on rows fit_idx, evaluated on eval_idx.
    X_t holds the day-t inputs in original units (factors..., lag1 as the last column).
        ols_factors       time-series factor regression on the factors and lag1
        ols_factors_only  the same without lag1
        hist_mean         mean excess return of the estimation window
    """
    return {
        'ols_factors': _ols_fit_predict(X_t[fit_idx], y[fit_idx], X_t[eval_idx]),
        'ols_factors_only': _ols_fit_predict(X_t[fit_idx, :-1], y[fit_idx], X_t[eval_idx, :-1]),
        'hist_mean': np.full(len(eval_idx), y[fit_idx].mean()),
    }


def summarize_fit(y: np.ndarray, yhat: np.ndarray, linear: np.ndarray) -> dict:
    """Fit metrics, improvement over the linear factor model and the DM test against it."""
    res = metrics(y, yhat)
    res['R2_vs_ols'] = float(1.0 - np.sum((y - yhat) ** 2) / np.sum((y - linear) ** 2))
    dm = dm_test(y - linear, y - yhat)   # positive: model fits better than OLS
    res['DM_vs_ols'] = dm['DM_stat']
    res['DM_vs_ols_p'] = dm['DM_pvalue']
    return res
