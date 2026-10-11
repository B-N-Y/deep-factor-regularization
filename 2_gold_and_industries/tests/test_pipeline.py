#!/usr/bin/env python
"""
Pipeline checks (gold and industry portfolios). Run before every batch of jobs:
    python tests/test_pipeline.py
Works with or without pytest.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ASSETS, EXPERIMENTS, WINDOW  # noqa: E402
from src.data import (assert_target_not_in_inputs, build_dataset, input_cols,  # noqa: E402
                      load_raw, prepare_splits)
from src.evaluation import benchmark_fits, dm_test, metrics  # noqa: E402
from src.models import DeepFactorModel  # noqa: E402


# ── data ────────────────────────────────────────────────────────────────────

def test_own_return_never_in_inputs():
    for exp in EXPERIMENTS:
        assert assert_target_not_in_inputs(exp, n_dates=15) == 15


def test_check_catches_own_return_in_inputs():
    """Negative control: a dataset whose inputs contain r_t must be rejected."""
    import src.data as D
    real = D._full_frame

    def leaky(asset, raw):
        out = real(asset, raw)
        out[ASSETS[asset]['factors'][0]] = out['target']       # replace a factor by the target itself
        return out
    D._full_frame = leaky
    try:
        assert_target_not_in_inputs('gold_2010_2025', n_dates=5)
    except AssertionError:
        return
    finally:
        D._full_frame = real
    raise AssertionError("own-return check did not detect r_t inside the inputs")


def test_window_layout_and_values():
    asset = 'gold'
    raw = load_raw(asset).set_index('Date')
    df, X = build_dataset('gold_2015_2020')
    assert X.shape[1:] == (WINDOW, len(input_cols(asset)))
    exc = raw['ret'] - raw['rf']
    for i in [0, 100, len(df) - 1]:
        d = df['Date'][i]
        pos = raw.index.get_loc(d)
        assert np.allclose(X[i, -1, :-1], raw.loc[d, ASSETS[asset]['factors']].values.astype(float))  # factors of day t
        assert np.isclose(X[i, -1, -1], exc.iloc[pos - 1])                                            # lag1 = r_{t-1}
        assert np.allclose(X[i, 0, :-1], raw.iloc[pos - WINDOW + 1][ASSETS[asset]['factors']].values.astype(float))
        assert np.isclose(df['target'][i], exc.iloc[pos])


def test_excess_returns_and_linear_benchmark():
    df, X = build_dataset('ind_hitec_2010_2025')
    data = prepare_splits(df, X)
    fit = np.concatenate([data['train']['idx'], data['val']['idx']])
    b = benchmark_fits(X[:, -1, :], df['target'].values, fit, data['test']['idx'])
    r2 = metrics(data['test']['y'], b['ols_factors'])['R2']
    assert r2 > 0.7, f"FF5+MOM should explain most of HiTec excess returns, got R2={r2:.2f}"


def test_splits_scaled_on_train_only():
    df, X = build_dataset('gold_2010_2025')
    data = prepare_splits(df, X)
    tr = data['train']['idx']
    assert tr.max() < data['val']['idx'].min() < data['test']['idx'].min()
    assert np.isclose(data['scaler_y'].center_[0], np.median(df['target'].values[tr]))


def test_industry_inputs_include_macro_block():
    from src.config import FF_FACTORS, MACRO_FACTORS
    df, X = build_dataset('ind_enrgy_2010_2025')
    assert input_cols('ind_enrgy') == FF_FACTORS + MACRO_FACTORS + ['lag1']
    assert X.shape[2] == len(FF_FACTORS) + len(MACRO_FACTORS) + 1
    # macro changes are on the industry (CRSP) calendar: identical to gold's on common days
    g = load_raw('gold').set_index('Date')
    i = load_raw('ind_enrgy').set_index('Date')
    common = g.index.intersection(i.index)[200:1200]
    same = (np.isclose(g.loc[common, 'dlog_dxy'], i.loc[common, 'dlog_dxy'])).mean()
    assert same > 0.95, f"macro block misaligned between calendars ({same:.2f})"


def test_samples_cover_2010_2025():
    for exp in ['gold_2010_2025', 'ind_utils_2010_2025']:
        df, _ = build_dataset(exp)
        assert str(df['Date'].min().date()) <= '2010-01-05' and str(df['Date'].max().date()) >= '2025-12-30'


# ── model ───────────────────────────────────────────────────────────────────

def _model(arch, dt, rate=0.3, alpha=5.0, dm='tv'):
    torch.manual_seed(0)
    return DeepFactorModel(arch, 7, 32, 3, dt, rate, alpha, dm)


def _inputs():
    torch.manual_seed(1)
    return torch.randn(64, WINDOW, 7)


def test_dropout_behaviour_all_archs():
    x = _inputs()
    for arch in ['lstm', 'gru', 'mlp']:
        m = _model(arch, 'none'); m.train()
        assert torch.allclose(m(x)[0], m(x)[0]), f"{arch}/none must be deterministic"
        for dt in ['fixed', 'dynamic', 'adaptive', 'monte_carlo', 'standard']:
            m = _model(arch, dt); m.train()
            assert not torch.allclose(m(x)[0], m(x)[0]), f"{arch}/{dt} must be stochastic in training"
            m.eval()
            assert torch.allclose(m(x)[0], m(x)[0]), f"{arch}/{dt} must be deterministic at inference"
        m = _model(arch, 'monte_carlo'); m.eval()
        assert not torch.allclose(m(x, mc=True)[0], m(x, mc=True)[0]), f"{arch}: MC must sample when mc=True"


def test_mlp_uses_only_day_t():
    m = _model('mlp', 'none'); m.eval()
    x = _inputs(); x2 = x.clone(); x2[:, :-1, :] += 10.0
    assert torch.allclose(m(x)[0], m(x2)[0])


def test_penalty_and_rates():
    x = _inputs()
    for arch in ['lstm', 'mlp']:
        for dt in ['fixed', 'dynamic', 'adaptive']:
            m = _model(arch, dt); m.train()
            assert m(x)[1].mean().item() > 0, f"{arch}/{dt}: divergence should be positive in training"
            m.eval()
            assert m(x)[1].abs().sum().item() == 0
        m = _model(arch, 'adaptive'); m.train(); m(x)
        assert any(abs(r - 0.3) > 1e-6 for r in m.current_rates()), f"{arch}: adaptive rates never move"
    m = _model('gru', 'dynamic'); m.update_dynamic_rate(0.02)
    assert np.isclose(m.dynamic_rate, min(0.3 + 5.0 * 0.02, 0.5))
    assert _model('lstm', 'fixed', rate=0.17).current_rates() == [0.17] * 3, "fixed must use the tuned rate"
    assert _model('lstm', 'monte_carlo').alpha == 0.0, "MC has no divergence penalty"
    m = _model('lstm', 'standard', rate=0.2); m.train()
    assert m.alpha == 0.0 and m(x)[1].abs().sum().item() == 0, "standard dropout has no penalty"
    assert m.current_rates() == [0.2] * 3, "standard dropout must use the tuned rate"


def test_nonfinite_loss_is_caught():
    from src.training import NonFiniteLoss, train_model
    m = _model('lstm', 'dynamic')
    m.compute_loss = lambda *a: torch.tensor(float('nan'))
    before = [p.detach().clone() for p in m.parameters()]
    try:
        train_model(m, _inputs().numpy(), np.zeros(64), 1, 'cpu')
    except NonFiniteLoss:
        assert all(torch.equal(a, b) for a, b in zip(before, m.parameters())), "parameters changed"
        return
    raise AssertionError("NaN loss was not caught")


def test_failed_trial_is_skipped():
    import run_grid_search as rgs
    from src.training import NonFiniteLoss
    df, X = build_dataset('gold_2015_2020')
    data = prepare_splits(df, X)
    real_train, calls = rgs.train_model, {'n': 0}

    def flaky(*a, **k):
        calls['n'] += 1
        if calls['n'] == 2:
            raise NonFiniteLoss("injected")
        return real_train(*a, **k)
    rgs.train_model = flaky
    try:
        _, rows, best = rgs.run_search('mlp', 'fixed', 'tv', data, n_trials=3, epochs=1)
    finally:
        rgs.train_model = real_train
    states = [r['State'] for r in rows]
    assert states.count('FAIL') == 1 and states.count('COMPLETE') == 2 and best is not None


def test_log_alpha_search_space():
    import optuna
    import run_grid_search as rgs
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    vals = []
    for _ in range(30):
        t = study.ask()
        vals.append(rgs.suggest_params(t, 'fixed', 'log')['alpha'])
        study.tell(t, 0.0)
    assert min(vals) >= 1e-3 and max(vals) <= 1e3 and max(vals) / min(vals) > 100
    assert ('standard', 'none') in rgs.make_combos() and len(rgs.make_combos()) == 47


def test_alternative_regularizers():
    """Weight decay shrinks the weights; early stopping stops on the validation set and restores the best epoch."""
    from src.training import build_model, train_model
    torch.manual_seed(2)
    X = torch.randn(256, WINDOW, 7).numpy()
    y = np.random.default_rng(0).normal(size=256)
    params = {'num_layers': 2, 'hidden_size': 32}
    norms = {}
    for wd in [0.0, 0.05]:
        torch.manual_seed(3)
        m = build_model('mlp', dict(params, weight_decay=wd), 'weight_decay', 'none', 7, 'cpu')
        m.train()
        assert torch.allclose(m(torch.tensor(X[:8]))[0], m(torch.tensor(X[:8]))[0]), "no dropout in weight_decay"
        train_model(m, X, y, 15, 'cpu', weight_decay=wd)
        norms[wd] = sum(float((p_ ** 2).sum()) for p_ in m.parameters())
    assert norms[0.05] < norms[0.0], "weight decay should shrink the weights"
    torch.manual_seed(4)
    m = build_model('lstm', params, 'early_stopping', 'none', 7, 'cpu')
    yv = np.random.default_rng(1).normal(size=64)
    hist = train_model(m, X, y, 30, 'cpu', early_stop={'X': X[:64], 'y_s': yv, 'patience': 3})
    assert 1 <= hist['best_epoch'] <= len(hist['loss']) <= 30
    assert len(hist['loss']) < 30 or hist['best_epoch'] > 27, "early stopping never triggered"


# ── statistics ──────────────────────────────────────────────────────────────

def test_dm_test():
    rng = np.random.default_rng(0)
    e1 = rng.normal(size=500)
    e2 = 0.8 * e1 + 0.1 * rng.normal(size=500)
    res = dm_test(e1, e2)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    manual = d.mean() / np.sqrt(d.var() / n) * np.sqrt((n - 1) / n)
    assert np.isclose(res['DM_stat'], manual) and res['DM_stat'] > 0
    assert np.isnan(dm_test(e1, e1)['DM_stat'])


def test_total_r2():
    y = np.array([1.0, -1.0, 2.0, 0.0]); yhat = np.zeros(4)
    assert np.isclose(metrics(y, yhat)['R2_total'], 0.0)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
