#!/usr/bin/env python
"""
make_tables.py
--------------
Every table, figure and number reported in the paper, built from the result folders of both
samples (read only):

    1_commodity_futures      results, results_logalpha, results_altreg, results_seed,
                             results_expanding_cv; data/ (built data and the LSEG downloads)
    2_gold_and_industries    results, results_logalpha, results_altreg, results_seed,
                             results_expanding_cv, results_timing; data/

The paper is organised around three hypotheses:
    H1  without regularization, deep factor models explain less than the linear factor model
        out of sample, and regularization restores their explanatory power
    H2  the regularization a network needs is larger where it overfits more: at an equal search
        budget, the gain of strong regularizers (weight decay, early stopping, a first-order
        consistency penalty) over standard dropout is larger in the networks that overfit most,
        first-order divergences (Proposition 1) win more often there, and their advantage shrinks
        when the penalty weight is searched on a log scale
    H3  the gains of deep models over the linear model arise where the linear factor loadings
        shift, and mostly reflect the shrinkage that regularization implies; diagnostics from
        loading_instability.py and h3_benchmarks.py (ridge, rolling regression, Chow test,
        tracking share); divergence_scale.py gives the size of the 14 penalties (Table C5)

Writes into --out: numbers.tex (every number quoted in the text, as LaTeX macros),
tables/*.tex, figures/*.pdf and MANIFEST.json.

    python 3_analysis/make_tables.py --out output
"""
import argparse
import glob
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest, friedmanchisquare, spearmanr, wilcoxon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FUT_DIR = os.path.join(os.path.dirname(HERE), '1_commodity_futures')
EQ_DIR = os.path.join(os.path.dirname(HERE), '2_gold_and_industries')
sys.path.insert(0, FUT_DIR)
from src.config import COMMODITIES, SECTORS  # noqa: E402
from src.evaluation import dm_test  # noqa: E402

FUT, EQ = 'Commodity futures', 'Gold and industry portfolios'
STUDIES = {FUT: FUT_DIR, EQ: EQ_DIR}
ARCH = ['lstm', 'gru', 'mlp']
ARCH_LABEL = {'lstm': 'LSTM', 'gru': 'GRU', 'mlp': 'MLP'}
PENALIZED = ['fixed', 'dynamic', 'adaptive']
FIRST_ORDER = {'wasserstein', 'tv', 'hellinger'}
CHANCE = 3 / 14
DIVERGENCES = ['kl', 'reverse_kl', 'js', 'pearson_chi_square', 'renyi', 'jensen_renyi', 'bhattacharyya',
               'tsallis', 'bregman_euclidean', 'bregman_exponential', 'bregman_log', 'tv', 'wasserstein', 'hellinger']
INDUSTRY = {'nodur': 'Consumer non-durables', 'durbl': 'Consumer durables', 'manuf': 'Manufacturing',
            'enrgy': 'Energy', 'hitec': 'High technology', 'telcm': 'Telecommunications', 'shops': 'Retail',
            'hlth': 'Health', 'utils': 'Utilities', 'other': 'Other'}
SECTOR_OF = {r.lower(): s for s, roots in SECTORS.items() for r in roots}
COLOR = {FUT: '#0072B2', EQ: '#D55E00'}           # Okabe-Ito blue / vermillion (colour-blind safe)
# Versions of the source files recorded by the runs reported in the paper. They differ only in regimes that are
# switched off by default (the outputs of every other regime are identical) and in comments. Results produced with
# the code of this repository carry a single version of each file.
COMPATIBLE = {'src/config.py': {'fb17b9bf9cf96697c6276f6b636b31e0', 'e7613a54f8396f67b52bc38396e66df8',
                                'de7db2454bced82012a8c5c8fa4c6989',
                                '5c4189966d908a5f598f84c5c1699ece', '2398f62e4fe09f92efa33fb23839e653'},
              'src/models.py': {'8230a66da5efe12b6d415c4354b2c11f', '0411d666df05c452b1d26e2633cb46ed'},
              'src/training.py': {'236e134ec331e4b61989e2e41c806dbb', 'b017482b0c6c367d35b94f702f0e24e6'}}
ALT = {'weight_decay': 'wd', 'early_stopping': 'es'}


# row order of the asset tables: the futures by sector as in Table B1, gold, then the industries in the order of the
# French Data Library
ASSET_ORDER = [r.lower() for roots in SECTORS.values() for r in roots] + ['gold'] + [f'ind_{k}' for k in INDUSTRY]


def asset_label(study, asset):
    if study == FUT:
        return COMMODITIES[asset.upper()]
    return 'Gold (GLD)' if asset == 'gold' else INDUSTRY[asset.replace('ind_', '')]


def split_exp(exp):
    parts = exp.split('_')
    return '_'.join(parts[:-2]), f'{parts[-2]}_{parts[-1]}'


# ---------------------------------------------------------------------------- provenance
def check_code(study, folders):
    """All result folders of a study must come from the same code (up to COMPATIBLE versions)."""
    seen = {}
    for d in folders:
        info = json.load(open(os.path.join(d, 'run_info.json')))
        for f, md5 in info.get('code_md5', {}).items():
            if f.startswith('src/') and f != 'src/utils.py':
                seen.setdefault(f, set()).add(md5)
    for f, md5s in seen.items():
        if len(md5s) > 1 and not md5s <= COMPATIBLE.get(f, set()):
            sys.exit(f"❌ {study}: results come from different versions of {f}: {md5s}")
    return {f: sorted(m) for f, m in seen.items()}


def finished(root):
    return sorted(d for d in glob.glob(os.path.join(root, '*', '*')) if '__part_' not in d
                  and os.path.isfile(os.path.join(d, 'run_info.json')))


# ---------------------------------------------------------------------------- loading
def test_errors(pred, key):
    y = np.asarray(pred['meta']['y_test'], float)
    entry = pred['models'][key] if key in pred['models'] else pred['benchmarks'][key]
    yhat = np.asarray(entry['test'] if isinstance(entry, dict) else entry, float)
    return y - yhat


def dm_nw(e1, e2):
    """Diebold-Mariano statistic on squared errors with a Newey-West (Bartlett) long-run variance,
    lag length floor(4 (n/100)^(2/9)); positive values favor the second model."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    dc = d - d.mean()
    lrv = dc @ dc / n + 2 * sum((1 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, lags + 1))
    return float(d.mean() / np.sqrt(lrv / n)), lags


def load_grid(study):
    """One row per experiment (main sample and subperiods) and one row per regime block."""
    base = STUDIES[study]
    rows, blocks, folders = [], [], finished(os.path.join(base, 'results'))
    for d in folders:
        model, exp = d.split(os.sep)[-2:]
        asset, period = split_exp(exp)
        b = pd.read_csv(os.path.join(d, 'best_per_combo.csv'))
        sel = json.load(open(os.path.join(d, 'selection.json')))['best_dropout']
        r = b[(b.Dropout_Type == sel['Dropout_Type']) & (b.Divergence == sel['Divergence'])].iloc[0]
        none = b[b.Dropout_Type == 'none'].iloc[0]
        bm = pd.read_csv(os.path.join(d, 'benchmarks.csv')).set_index('Benchmark')
        pred = json.load(open(os.path.join(d, 'predictions.json')))
        dm_none = dm_test(test_errors(pred, 'none__none'), test_errors(pred, f"{sel['Dropout_Type']}__{sel['Divergence']}"))
        dm_ols_nw, nw_lags = dm_nw(test_errors(pred, 'ols_factors'), test_errors(pred, f"{sel['Dropout_Type']}__{sel['Divergence']}"))
        dm_ols_abs = dm_test(test_errors(pred, 'ols_factors'), test_errors(pred, f"{sel['Dropout_Type']}__{sel['Divergence']}"),
                             power=1)['DM_stat']
        std = b[b.Dropout_Type == 'standard']
        if not len(std):
            sd = os.path.join(base, 'results_standard', model, exp, 'best_per_combo.csv')
            std = pd.read_csv(sd) if os.path.isfile(sd) else std   # standard dropout searched in a separate run
        std_r2 = float(std.Test_R2.iloc[0]) if len(std) else np.nan
        std_rate = float(std.Base_Rate.iloc[0]) if len(std) else np.nan
        std_val = float(std.Val_RMSE.iloc[0]) if len(std) else np.nan
        std_train = float(std.Train_R2.iloc[0]) if len(std) else np.nan
        regime_r2 = {f'r2_{dt}': float(b[b.Dropout_Type == dt].sort_values('Val_RMSE').Test_R2.iloc[0])
                     for dt in ['monte_carlo'] + PENALIZED if (b.Dropout_Type == dt).any()}
        # equal search budget: every penalized combination had 50 trials, like standard dropout
        pen = b[b.Dropout_Type.isin(PENALIZED)]
        first = pen.Divergence.isin(FIRST_ORDER)
        regime_r2.update(eq_gain=float((pen.Test_R2 - std_r2).median()),
                         eq_share=float((pen.Test_R2 > std_r2).mean()),
                         fo_gain=float((pen[first].Test_R2 - std_r2).median()),
                         so_gain=float((pen[~first].Test_R2 - std_r2).median()),
                         r2_pen_median=float(pen.Test_R2.median()),
                         dm_ols_p=float(r.Test_DM_vs_ols_p), std_val=std_val, dm_ols_nw=dm_ols_nw, nw_lags=nw_lags,
                         dm_ols_abs=dm_ols_abs, std_rate=std_rate, std_train=std_train)
        # comparison regimes without dropout, same search budget as standard dropout (results_altreg)
        ad = os.path.join(base, 'results_altreg', model, exp)
        if os.path.isfile(os.path.join(ad, 'run_info.json')):
            alt = pd.read_csv(os.path.join(ad, 'best_per_combo.csv'))
            for dt, k in ALT.items():
                a = alt[alt.Dropout_Type == dt].iloc[0]
                regime_r2.update({k: float(a.Test_R2), f'{k}_val': float(a.Val_RMSE), f'{k}_train': float(a.Train_R2),
                                  f'{k}_dm': float(a.Test_DM_vs_ols)})
            # size of the weight decay fit relative to the linear model's (near zero: the network collapsed to a constant)
            fit_wd = json.load(open(os.path.join(ad, 'predictions.json')))['models']['weight_decay__none']['test']
            regime_r2['wd_sdratio'] = float(np.std(fit_wd) / np.std(pred['benchmarks']['ols_factors']))
            regime_r2.update(wd_lambda=float(alt[alt.Dropout_Type == 'weight_decay'].Weight_Decay.iloc[0]),
                             es_epoch=float(alt[alt.Dropout_Type == 'early_stopping'].ES_Epochs.iloc[0]),
                             fo_r2=float(pen[first].Test_R2.median()))
        rows.append(dict(**regime_r2, study=study, model=model, asset=asset, period=period,
                         ols=float(bm.loc['ols_factors', 'R2']), none=float(none.Test_R2),
                         none_train=float(none.Train_R2), sel=float(r.Test_R2), sel_train=float(r.Train_R2),
                         sel_val=float(r.Val_R2), sel_total=float(r.Test_R2_total), none_val=float(none.Val_R2),
                         regime=sel['Dropout_Type'], div=sel['Divergence'], alpha=float(r.Alpha),
                         dm_ols=float(r.Test_DM_vs_ols), dm_none=float(dm_none['DM_stat']), standard=std_r2,
                        penalized=sel['Dropout_Type'] in PENALIZED))
        for dt in PENALIZED:
            g = b[b.Dropout_Type == dt]
            if len(g) == len(DIVERGENCES):
                blocks.append(dict(study=study, model=model, asset=asset, period=period, regime=dt,
                                   win=g.sort_values('Val_RMSE').Divergence.iloc[0], scale='linear'))
    la_folders = finished(os.path.join(base, 'results_logalpha'))
    la_rows = []
    for d in la_folders:
        model, exp = d.split(os.sep)[-2:]
        asset, period = split_exp(exp)
        b = pd.read_csv(os.path.join(d, 'best_per_combo.csv'))
        sel = json.load(open(os.path.join(d, 'selection.json')))['best_dropout']
        r = b[(b.Dropout_Type == sel['Dropout_Type']) & (b.Divergence == sel['Divergence'])].iloc[0]
        la_rows.append(dict(study=study, model=model, asset=asset, period=period, sel_log=float(r.Test_R2),
                            alpha_log=float(r.Alpha), penalized_log=sel['Dropout_Type'] in PENALIZED))
        for dt in PENALIZED:
            g = b[b.Dropout_Type == dt]
            if len(g) == len(DIVERGENCES):
                blocks.append(dict(study=study, model=model, asset=asset, period=period, regime=dt,
                                   win=g.sort_values('Val_RMSE').Divergence.iloc[0], scale='log'))
    alt_folders = finished(os.path.join(base, 'results_altreg'))
    return pd.DataFrame(rows), pd.DataFrame(blocks), pd.DataFrame(la_rows), folders + la_folders + alt_folders


# On the probability simplex the logarithmic Bregman divergence equals the Kullback-Leibler divergence and the
# Tsallis divergence of order 2 equals the Pearson chi-square divergence; the other second-order penalties differ.
DUPLICATES = [('kl', 'bregman_log'), ('pearson_chi_square', 'tsallis')]
SECOND_ORDER_DISTINCT = ['kl', 'reverse_kl', 'js', 'pearson_chi_square', 'jensen_renyi', 'bhattacharyya',
                         'bregman_euclidean', 'bregman_exponential']


def duplicate_placebo(study):
    """Main-sample regime blocks: absolute difference in test R2 between the searches of the two identical penalties
    of each pair, against that between distinct second-order penalties, and the share of pairs in which the two
    searches select the same configuration (their configurations and seeds coincide trial by trial)."""
    dup, dist, same = [], [], []
    for d in finished(os.path.join(STUDIES[study], 'results')):
        if not d.endswith('2010_2025'):
            continue
        b = pd.read_csv(os.path.join(d, 'best_per_combo.csv'))
        for dt in PENALIZED:
            g = b[b.Dropout_Type == dt].set_index('Divergence')
            if len(g) != len(DIVERGENCES):
                continue
            for x, y in DUPLICATES:
                dup.append(abs(g.Test_R2[x] - g.Test_R2[y]))
                same.append(g.Best_Trial[x] == g.Best_Trial[y])
            dist += [abs(g.Test_R2[x] - g.Test_R2[y]) for x, y in itertools.combinations(SECOND_ORDER_DISTINCT, 2)]
    return dict(dup=float(np.median(dup)), dist=float(np.median(dist)), same=float(np.mean(same)), n=len(dup))


def load_instability(study):
    out = subprocess.run([sys.executable, os.path.join(HERE, 'loading_instability.py'), STUDIES[study]],
                         capture_output=True, text=True, check=True).stdout
    return pd.DataFrame(json.loads(out.strip().splitlines()[-1])).assign(study=study)


def load_h3bench(study):
    out = subprocess.run([sys.executable, os.path.join(HERE, 'h3_benchmarks.py'), STUDIES[study]],
                         capture_output=True, text=True, check=True).stdout
    return pd.DataFrame(json.loads(out.strip().splitlines()[-1])).assign(study=study)


def load_seeds(study):
    out = []
    for f in glob.glob(os.path.join(STUDIES[study], 'results_seed', '*', '*', 'seed_*', 'seed_result.csv')):
        model, exp, seed = f.split(os.sep)[-4:-1]
        for _, r in pd.read_csv(f).iterrows():
            out.append(dict(study=study, model=model, asset=split_exp(exp)[0], seed=seed,
                            kind='none' if r.Dropout_Type == 'none' else 'dropout', r2=r.Test_R2))
    return pd.DataFrame(out)


def load_cv(study):
    out = []
    for f in glob.glob(os.path.join(STUDIES[study], 'results_expanding_cv', '*', '*', 'fold_results.csv')):
        model, exp = f.split(os.sep)[-3:-1]
        c = pd.read_csv(f)
        c = c[~c.Model.astype(str).str.startswith('DM ')]
        for fold, g in c.groupby('Fold'):
            r2 = g.set_index('Model').R2
            sel = [k for k in r2.index if not k.startswith('benchmark') and k != 'none__none']
            if not sel or 'none__none' not in r2.index or 'benchmark:ols_factors' not in r2.index:
                continue
            out.append(dict(study=study, model=model, asset=split_exp(exp)[0], fold=fold, sel=r2[sel[0]],
                            none=r2['none__none'], ols=r2['benchmark:ols_factors']))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- formatting helpers
def f3(x):
    return '--' if pd.isna(x) else f'{x + 0.0:.3f}'.replace('-0.000', '0.000').replace('-', '$-$')


def sci(x):
    """Scientific notation as in print: 2.13e-02 -> 2.13 x 10^-2 (plain when the exponent is zero)."""
    m, e = f'{x:.2e}'.split('e')
    return m if int(e) == 0 else f'${m}\\times 10^{{{int(e)}}}$'


def pct(x):
    return f'{100 * x:.0f}'


def pval(p):
    if pd.isna(p):
        return '--'
    return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def table(caption, label, colspec, header, body, notes, wide=False):
    env = 'table'
    return (f"\\begin{{{env}}}[ht]\n\\centering\n\\caption{{{caption}}}\\label{{{label}}}\n\\footnotesize\n"
            f"\\begin{{adjustbox}}{{max width=\\linewidth}}\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
            + '\n'.join(body) + f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{adjustbox}}\n"
            f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\scriptsize\\emph{{Notes:}} {notes}}}\n\\end{{{env}}}\n")


def wilcox_p(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return wilcoxon(a, b).pvalue if len(a) > 5 and np.any(a != b) else np.nan


# ---------------------------------------------------------------------------- tables
def tab_desc(out):
    d = pd.read_csv(os.path.join(FUT_DIR, 'data', 'futures_data.csv'), parse_dates=['Date']).set_index('Date').loc['2010':]
    rows = []
    for sector, roots in SECTORS.items():
        r = d[[f'ret_{x}' for x in roots]]
        c = r.corr().values
        rows.append(f"{sector.capitalize()} & {len(roots)} & {f3(np.mean(r.mean() * 252) / 100)} & "
                    f"{f3(np.mean(r.std() * np.sqrt(252)) / 100)} & {f3(c[np.triu_indices(len(roots), 1)].mean())} \\\\")
    fac = []
    for col, lab in [('avg', 'Commodity market (AVG)'), ('carry', 'Carry (CARRY)'), ('mom', 'Momentum (MOM)'),
                     ('mkt_rf', 'Equity market excess return')]:
        x = d[col]
        fac.append(f"{lab} & & {f3(x.mean() * 252 / 100)} & {f3(x.std() * np.sqrt(252) / 100)} & \\\\")
    g = pd.read_csv(os.path.join(EQ_DIR, 'data', 'gold_factors.csv'), parse_dates=['Date']).set_index('Date').loc['2010':]
    ind = pd.read_csv(os.path.join(EQ_DIR, 'data', 'industry_factors.csv'), parse_dates=['Date']).set_index('Date').loc['2010':]
    gx = g.gold_ret - g.rf
    eq = [f"Gold (GLD) & 1 & {f3(gx.mean() * 252 / 100)} & {f3(gx.std() * np.sqrt(252) / 100)} & \\\\"]
    cols = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth', 'Utils', 'Other']
    ix = ind[cols].sub(ind.rf, axis=0)
    c = ix.corr().values
    eq.append(f"Industry portfolios & 10 & {f3((ix.mean() * 252 / 100).mean())} & "
              f"{f3((ix.std() * np.sqrt(252) / 100).mean())} & {f3(c[np.triu_indices(10, 1)].mean())} \\\\")
    body = (["\\multicolumn{5}{l}{\\emph{Panel A: commodity futures by sector}} \\\\"] + rows +
            ["\\multicolumn{5}{l}{\\emph{Panel B: commodity and equity market factors}} \\\\"] + fac +
            ["\\multicolumn{5}{l}{\\emph{Panel C: gold and industry portfolios}} \\\\"] + eq)
    write(os.path.join(out, 'tables', 'tab_desc.tex'), table(
        'Descriptive statistics, January 2010 to December 2025', 'tab:desc', 'lcccc',
        'Group & Assets & Mean return & Volatility & Mean correlation', body,
        'Annualized mean and volatility of daily returns (decimal), averaged over the assets in a group; mean '
        'correlation is the average pairwise correlation of daily returns within the group. Futures returns are '
        'excess returns of the held contract; gold and industry returns are in excess of the T-bill rate. Panel B: the '
        'commodity factors built from all 26 commodities; the leave-one-out versions used as inputs differ only marginally.'))


def tab_divergences(out):
    rows = [('Kullback--Leibler', r'$\sum_k p_k\log(p_k/q_k)$', 2), ('Reverse Kullback--Leibler', r'$\sum_k q_k\log(q_k/p_k)$', 2),
            ('Jensen--Shannon', r'$\tfrac12 D_{KL}(p\|m)+\tfrac12 D_{KL}(q\|m)$, $m=\tfrac{p+q}{2}$', 2),
            (r'Pearson $\chi^2$', r'$\sum_k (p_k-q_k)^2/q_k$', 2), (r'R\'enyi (order 2)', r'$\log\sum_k p_k^2/q_k$', 2),
            (r'Jensen--R\'enyi', r'$\tfrac12\,[D_{JS}+D_{R,2}]$', 2), ('Bhattacharyya', r'$-\log\sum_k\sqrt{p_kq_k}$', 2),
            ('Tsallis (order 2)', r'$\sum_k p_k^2/q_k-1$', 2), ('Bregman, Euclidean', r'$\sum_k (p_k-q_k)^2$', 2),
            ('Bregman, exponential', r'$\sum_k [e^{p_k}-e^{q_k}-(p_k-q_k)e^{q_k}]$', 2),
            ('Bregman, logarithmic', r'$\sum_k [p_k\log(p_k/q_k)-(p_k-q_k)]$', 2),
            ('Total variation', r'$\tfrac12\sum_k|p_k-q_k|$', 1), ('Wasserstein-1', r'$\sum_k|\sum_{j\le k}(p_j-q_j)|$', 1),
            ('Hellinger distance', r'$\tfrac{1}{\sqrt2}\|\sqrt p-\sqrt q\|_2$', 1)]
    body = [f"{n} & {f} & {'first' if o == 1 else 'second'} \\\\" for n, f, o in rows]
    write(os.path.join(out, 'tables', 'tab_divergences.tex'), table(
        'The 14 consistency penalties and their local order', 'tab:divergences', 'llc',
        'Divergence & $D(p,q)$ & Local order', body,
        r'$p=\mathrm{softmax}(\mathbf h)$, $q=\mathrm{softmax}(\tilde{\mathbf h})$. Local order: the power of '
        r'$\|q-p\|$ at which the divergence vanishes as the perturbation shrinks proportionally (Proposition~\ref{prop:order}; '
        r'Table~\ref{tab:app_divscale} for dropout perturbations). '
        r'On the probability simplex the logarithmic Bregman divergence equals the Kullback--Leibler divergence and the '
        r'Tsallis divergence equals the Pearson $\chi^2$ divergence (Section~\ref{sec:penalty}). '
        r'A floor of $10^{-10}$ guards logarithms and divisions.'))


def first_share(bl):
    k, n = int(bl.win.isin(FIRST_ORDER).sum()), len(bl)
    return k, n, (k / n if n else np.nan), (binomtest(k, n, CHANCE, alternative='greater').pvalue if n else np.nan)


def h3_stats(main):
    out = {}
    for name, g in [(FUT, main[main.study == FUT]), (EQ, main[main.study == EQ]), ('all', main)]:
        rho, p = spearmanr(g.instability, g.sel - g.ols)
        out[name] = (rho, p, len(g))
    return out


def tab_setup(out):
    regimes = [('No regularization', '--', 'no', 'deterministic'),
               ('Standard dropout', '$p_0$', 'no', 'deterministic'),
               ('Monte Carlo', '$p_0$', 'no', 'average of 50 dropout passes'),
               ('Fixed + penalty', '$p_0$', r'$\alpha D$', 'deterministic'),
               ('Dynamic + penalty', r'$\min\{p_0+\alpha\bar D,0.5\}$, shared by all layers', r'$\alpha D$', 'deterministic'),
               ('Adaptive + penalty', r'$\min\{p_0+\alpha D^{(\ell)},0.5\}$, per layer', r'$\alpha D$', 'deterministic')]
    body = [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in regimes]
    body += [r"\multicolumn{4}{l}{\emph{Comparison regimes, no dropout}} \\",
             r"Weight decay & -- & $\tfrac{\lambda}{2}\|\boldsymbol\theta\|_2^2$ & deterministic \\",
             r"Early stopping & -- & no (stops when the validation error stops falling) & deterministic \\"]
    search = [('Layers', r'$\{2,3,4\}$'), ('Units per layer', r'$\{128,256,512\}$'), ('Base rate $p_0$', r'$[0.01,0.50]$'),
              (r'Penalty weight $\alpha$', r'$\{0,0.5,\dots,10\}$; for H2(d): log-uniform on $[10^{-3},10^{3}]$'),
              (r'Weight decay $\lambda$', r"log-uniform on $[10^{-6},10^{-1}]$ (Adam's weight-decay option)"),
              ('Early stopping', 'patience of 5 epochs, best epoch restored; refit for 1.5 times the best epoch'),
              ('Search', '50 configurations per regime and divergence (TPE, seed 42); 20 epochs on the training block'),
              ('Selection and refit', 'lowest validation RMSE; refit for 30 epochs on training and validation blocks'),
              ('Optimizer', r'Adam, learning rate $10^{-3}$, mini-batch 64, gradient clipping at 5'),
              ('Inputs', r'20-day window (LSTM, GRU); day $t$ only (MLP); median/IQR scaling from the training block')]
    body += [r"\midrule", r"\multicolumn{4}{l}{\emph{Panel B: hyperparameters and estimation}} \\"]
    body += [f"{a} & \\multicolumn{{3}}{{l}}{{{b}}} \\\\" for a, b in search]
    write(os.path.join(out, 'tables', 'tab_setup.tex'), table(
        'Regularization regimes and estimation settings', 'tab:setup', 'llll',
        r'\multicolumn{4}{l}{\emph{Panel A: regularization regimes}} \\ Regime & Dropout rate & Penalty & At test time', body,
        r'$D$: divergence between the softmax of the final hidden state with and without dropout (Table~\ref{tab:divergences}); '
        r'$\bar D$: its mean over the previous mini-batch; $D^{(\ell)}$: the local divergence at layer $\ell$. '
        r'$\boldsymbol\theta$: all network parameters. '
        'The same settings are used for every test asset and architecture.'))


FRIEDMAN, VIF, COST = {}, {}, {}
# Panel A: the models whose training-block fit is recorded, training and test R^2 in adjacent rows
TRAIN_TEST_ROWS = [('ols_train', 'ols', 'Linear factor model'), ('none_train', 'none', 'No regularization'),
                   ('std_train', 'standard', 'Standard dropout'), ('wd_train', 'wd', 'Weight decay'),
                   ('es_train', 'es', 'Early stopping'), ('sel_train', 'sel', 'Selected dropout model')]
# Panel B: test R^2 of the other models
OTHER_ROWS = [('olswin_test', 'Linear model, 20-day window'), ('ridge_test', 'Ridge regression'),
              ('r2_monte_carlo', 'Monte Carlo'), ('r2_pen_median', 'Penalized, typical combination'),
              ('r2_fixed', 'Fixed + penalty'), ('r2_dynamic', 'Dynamic + penalty'), ('r2_adaptive', 'Adaptive + penalty')]


def linear_cells(cells):
    """The linear models do not depend on the network: one value per sample, in its All column."""
    assert len(set(cells[:4])) == 1 and len(set(cells[4:])) == 1, cells
    return all_only(cells[3], cells[7])


def all_only(fut, eq):
    """Cells of a row with one value per sample, placed in the All columns."""
    return f" & & & {fut} & & & & {eq}"


LINEAR_ROWS = ('ols', 'ols_train', 'olswin_test', 'ridge_test')


def tab_regimes(main, out):
    cols = [(st, a) for st in [FUT, EQ] for a in ARCH + ['all']]
    count_key = {'r2_monte_carlo': 'monte_carlo', 'r2_fixed': 'fixed', 'r2_dynamic': 'dynamic', 'r2_adaptive': 'adaptive'}

    def row(key):
        cells = []
        for st, a in cols:
            g = main[(main.study == st) & ((main.model == a) if a != 'all' else True)]
            cell = f3(g[key].median())
            if key in count_key:
                cell += f" ({int((g.regime == count_key[key]).sum())})"
            cells.append(cell)
        return linear_cells(cells) if key in LINEAR_ROWS else ' & '.join(cells)

    body = [r"\multicolumn{10}{l}{\emph{Panel A: training and test $R^2$}} \\"]
    for i, (train, test, lab) in enumerate(TRAIN_TEST_ROWS):
        if i:
            body.append('\\addlinespace')
        body.append(f"{lab} & Training & {row(train)} \\\\")
        body.append(f" & Test & {row(test)} \\\\")
    body.append('\\midrule')
    body.append(r"\multicolumn{10}{l}{\emph{Panel B: test $R^2$ of the other models}} \\")
    for key, lab in OTHER_ROWS:
        body.append(f"\\multicolumn{{2}}{{l}}{{{lab}}} & {row(key)} \\\\")
    body.append('\\midrule')
    for other, lab in [('ols', 'linear factor model'), ('none', 'no regularization'), ('standard', 'standard dropout'),
                       ('wd', 'weight decay'), ('es', 'early stopping')]:
        beats = []
        for st, a in cols:
            g = main[(main.study == st) & ((main.model == a) if a != 'all' else True)].dropna(subset=[other])
            beats.append(pct((g.sel > g[other]).mean()))
        body.append(f"\\multicolumn{{2}}{{l}}{{Selected $>$ {lab} (\\%)}} & " + ' & '.join(beats) + ' \\\\')
    fr = []
    for st in [FUT, EQ]:
        g = main[main.study == st][['r2_monte_carlo', 'r2_fixed', 'r2_dynamic', 'r2_adaptive']].dropna()
        fr.append(pval(friedmanchisquare(*[g[c] for c in g.columns]).pvalue))
    FRIEDMAN.update({FUT: fr[0], EQ: fr[1]})
    fa = []
    for st in [FUT, EQ]:
        g = main[main.study == st][['standard', 'wd', 'es']].dropna()
        fa.append(pval(friedmanchisquare(*[g[c] for c in g.columns]).pvalue))
    FRIEDMAN.update({FUT + ' alt': fa[0], EQ + ' alt': fa[1]})
    body.append(r"\midrule")
    body.append(f"\\multicolumn{{2}}{{l}}{{Friedman $p$, four dropout regimes}} &" + all_only(fr[0], fr[1]) + " \\\\")
    body.append(f"\\multicolumn{{2}}{{l}}{{Friedman $p$, three single-search regimes}} &" + all_only(fa[0], fa[1]) + " \\\\")
    header = (r" & & \multicolumn{4}{c}{Commodity futures} & \multicolumn{4}{c}{Gold and industry portfolios} \\"
              r" \cmidrule(lr){3-6}\cmidrule(lr){7-10} & & LSTM & GRU & MLP & All & LSTM & GRU & MLP & All")
    write(os.path.join(out, 'tables', 'tab_regimes.tex'), table(
        r'Median training and test $R^2$ by regime', 'tab:regimes', 'll' + 'c' * 8, header, body,
        'Medians over experiments: in the LSTM, GRU and MLP columns over the test assets (26 futures; gold and ten industry '
        'portfolios), in the All columns over test assets and networks. The linear models do not depend on the network, so '
        'their medians are shown only in the All columns. Training: $R^2$ on the training block of the configuration chosen for each regime, '
        'estimated on that block in the search (linear model: the regression estimated on the training block); test: $R^2$ '
        'on the test block after re-estimation on the training and validation blocks. Weight decay and early stopping: '
        'networks without dropout. Linear model, 20-day window: the linear regression on the 20-day input window of the '
        'recurrent networks. Ridge regression: penalty chosen on the validation block. Every search has 50 configurations: '
        'one search per regime, and one per divergence in the penalized regimes. Monte Carlo and penalized regimes: the '
        'configuration with the lowest validation error (penalized regimes: over their 14 searches); in parentheses, the '
        'number of experiments in which the regime provides the selected dropout model, the configuration with the lowest '
        'validation error over these 43 searches. Penalized, typical combination: median over the 42 penalized combinations '
        'of an experiment. Friedman $p$: Friedman test of equal test $R^2$ across regimes, with all experiments of a sample as blocks, '
        'for the four candidate dropout regimes (Monte Carlo, fixed, dynamic, adaptive) and for the three regimes with a '
        'single search (standard dropout, weight decay, early stopping).',
        wide=True))


def mcnemar_first_order(blocks, study):
    key = ['model', 'asset', 'period', 'regime']
    b = blocks[blocks.study == study]
    lin = b[b.scale == 'linear'].set_index(key).win.isin(FIRST_ORDER)
    log = b[b.scale == 'log'].set_index(key).win.isin(FIRST_ORDER)
    both = lin.to_frame('lin').join(log.to_frame('log'), how='inner')
    n10 = int((both.lin & ~both.log).sum())
    n01 = int((~both.lin & both.log).sum())
    return binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else np.nan


RECURRENT = ['lstm', 'gru']


def arch_pair(s_, gain):
    """Per test asset (and period), the mean gain in the two recurrent networks minus the gain in the MLP. The
    networks of an asset share its test returns, so the comparison is paired."""
    d = pd.DataFrame({'key': (s_.asset + '_' + s_.period).values, 'rec': s_.model.isin(RECURRENT).values,
                      'gain': np.asarray(gain, float)})
    return (d[d.rec].groupby('key').gain.mean() - d[~d.rec].groupby('key').gain.mean()).dropna()


def arch_gain(s_, gain):
    """Median gain in the recurrent networks / in the MLP, and the Wilcoxon signed-rank p-value of their difference,
    paired by test asset (arch_pair)."""
    rec, mlp = gain[s_.model.isin(RECURRENT)], gain[s_.model == 'mlp']
    return f"{f3(rec.median())} / {f3(mlp.median())}", pval(wilcoxon(arch_pair(s_, gain)).pvalue)


def gap_closed(g, col):
    """Median share of the test-block gap between the unregularized network and the linear model that a regularized
    network closes, over the experiments in which the unregularized network is below the linear model."""
    h = g[g.none < g.ols].dropna(subset=[col])
    return ((h[col] - h.none) / (h.ols - h.none)).median()


def test_summary(main, blocks, logsel):
    """Statistic and p-value of every hypothesis test, per sample."""
    res = {}
    for st in [FUT, EQ]:
        g = main[main.study == st]
        s_ = g.dropna(subset=['standard'])
        lin = blocks[(blocks.study == st) & (blocks.scale == 'linear')]
        log = blocks[(blocks.study == st) & (blocks.scale == 'log')]
        rho, p = spearmanr(g.ols, g.sel - g.ols)
        res[st] = {
            'h1a': (pct((g.none < g.ols).mean()) + '\\%', pval(wilcox_p(g.none, g.ols))),
            'h1b': (pct((g.sel > g.none).mean()) + '\\%', pval(wilcox_p(g.sel, g.none))),
            'h1c': (f"{pct((g.wd > g.none).mean())}\\,/\\,{pct((g.es > g.none).mean())}\\%",
                    pval(max(wilcox_p(g.wd, g.none), wilcox_p(g.es, g.none)))),
            'h1d': (f"{pct(gap_closed(g, 'sel'))}\\,/\\,{pct(gap_closed(g, 'wd'))}\\,/\\,{pct(gap_closed(g, 'es'))}\\%", '--'),
            'h2a_wd': arch_gain(s_, s_.wd - s_.standard),
            'h2a_es': arch_gain(s_, s_.es - s_.standard),
            'h2a': arch_gain(s_, s_.fo_gain),
            'h2a_half': (f3(s_[s_.shortfall > s_.shortfall.median()].fo_gain.median()),
                         pval(wilcoxon(s_[s_.shortfall > s_.shortfall.median()].fo_gain).pvalue)),
            'h2_eq': (pct(s_.eq_share.mean()) + '\\%', pval(wilcoxon(s_.eq_gain).pvalue)),
            'h2_sel': (pct((s_.sel > s_.standard).mean()) + '\\%', pval(wilcox_p(s_.sel, s_.standard))),
            'h2a_share': (pct((s_.sel > s_.standard).mean()) + '\\%', pval(wilcox_p(s_.sel, s_.standard))),
            'h2a_gap': (f3(spearmanr(s_.gap, s_.pen_gain)[0]), pval(spearmanr(s_.gap, s_.pen_gain)[1])),
            'h2c_rho': (f3(spearmanr(g.short_val, g.first_share)[0]), pval(spearmanr(g.short_val, g.first_share)[1])),
            'h2b': (pct(first_share(lin)[2]) + '\\%', pval(first_share(lin)[3])),
            'h2c': ((pct(first_share(log)[2]) + '\\%') if len(log) else '--', pval(mcnemar_first_order(blocks, st)) if len(log) else '--'),
            'h3': (f3(spearmanr(g.instability, g.sel - g.ols)[0]), pval(spearmanr(g.instability, g.sel - g.ols)[1])),
            'h3b': (f3(spearmanr(g.ridge_test - g.ols, g.sel - g.ols)[0]), pval(spearmanr(g.ridge_test - g.ols, g.sel - g.ols)[1])),
            'h3t': (f3(g.track_rest.median()), pval(wilcoxon(g.track_rest).pvalue)),
            'h3c': (f3(spearmanr(g.ex_ante, g.sel - g.ols)[0]), pval(spearmanr(g.ex_ante, g.sel - g.ols)[1])),
            'h3_r2': (f3(rho), pval(p)),
        }
    return res


def tab_tests(main, blocks, logsel, out):
    res = test_summary(main, blocks, logsel)
    rows = [('H1', 'Unregularized network below linear model', 'share of experiments, Wilcoxon', 'h1a'),
            ('', 'Dropout beats unregularized network', 'share, Wilcoxon', 'h1b'),
            ('', 'Weight decay / early stopping beat it', 'shares, Wilcoxon', 'h1c'),
            ('', 'Regularization closes most of the gap: dropout / WD / ES', 'median share of gap closed', 'h1d'),
            ('H2a', 'Gain over standard dropout larger in the networks that overfit most: weight decay', 'recurrent / MLP, paired Wilcoxon', 'h2a_wd'),
            ('', '\\ldots early stopping', 'recurrent / MLP, paired Wilcoxon', 'h2a_es'),
            ('', '\\ldots first-order penalty', 'recurrent / MLP, paired Wilcoxon', 'h2a'),
            ('H2b', 'First-order divergences win, bounded $\\alpha$', 'share vs 21\\%, binomial', 'h2b'),
            ('H2c', '\\ldots more often where overfitting is severe', 'Spearman $\\rho$', 'h2c_rho'),
            ('H2d', '\\ldots less often with log-scale $\\alpha$', 'share, McNemar', 'h2c'),
            ('H3a', 'Gain over linear model rises with instability (ex post)', 'Spearman $\\rho$', 'h3'),
            ('H3b', '\\ldots and with the gain of a ridge regression', 'Spearman $\\rho$', 'h3b'),
            ('H3c', 'Network follows the shift in loadings beyond shrinkage', 'median share, Wilcoxon', 'h3t'),
            ('H3d', 'Gain not positively related to instability before the test block', 'Spearman $\\rho$', 'h3c')]
    body = []
    for h, d, m, k in rows:
        if h in ('H2a', 'H3a'):                      # space between the three hypotheses
            body.append('\\addlinespace')
        body.append(f"{h} & {d} & {m} & {res[FUT][k][0]} & {res[FUT][k][1]} & {res[EQ][k][0]} & {res[EQ][k][1]} \\\\")
    header = (r" & & & \multicolumn{2}{c}{Commodity futures} & \multicolumn{2}{c}{Gold and industry portfolios} \\"
              r" \cmidrule(lr){4-5}\cmidrule(lr){6-7} & Prediction & Test & Statistic & $p$ & Statistic & $p$")
    write(os.path.join(out, 'tables', 'tab_tests.tex'), table(
        'Summary of hypothesis tests', 'tab:tests', 'l>{\\raggedright\\arraybackslash}p{5.2cm}lcccc', header, body,
        'The hypotheses, measures and tests are defined in Section~\\ref{sec:hypotheses}. Shares are of experiments (H1) '
        'or of regime blocks (H2b, H2d); for H2a, the statistic is the median gain in test $R^2$ over standard dropout in '
        'the recurrent networks / in the MLP. H1, third row: the larger of the two $p$-values. WD: weight decay; ES: early '
        'stopping.',
        wide=True))
    sig = main[main.dm_ols > 1.96].assign(_a=lambda x: x.asset.map(ASSET_ORDER.index),
                                          _m=lambda x: x.model.map(ARCH.index)).sort_values(['study', '_a', '_m'])
    regime_name = {'monte_carlo': 'Monte Carlo', 'fixed': 'Fixed', 'dynamic': 'Dynamic', 'adaptive': 'Adaptive'}
    body = []
    for _, r in sig.iterrows():
        if not body or r.study != prev:              # one block per sample, as in the other tables
            body.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{r.study}}}}} \\\\")
            prev = r.study
        mark = '$^{*}$' if r.bh else ''
        dm_ridge = f"{r.dm_sel_ridge:.2f}".replace('-', '$-$')
        body.append(f"{asset_label(r.study, r.asset)} & {ARCH_LABEL[r.model]} & {f3(r.instability)} & {f3(r.ols)} & "
                    f"{f3(r.ridge_test)} & {f3(r.sel)} & {r.dm_ols:.2f}{mark} & {r.dm_ols_nw:.2f} & {dm_ridge} & "
                    f"{regime_name[r.regime]}{'' if r.regime == 'monte_carlo' else ', ' + DIV_LABEL[r['div']]} \\\\")
    write(os.path.join(out, 'tables', 'tab_sig.tex'), table(
        'Significant gains over the linear factor model', 'tab:sig', 'llcccccccl',
        r'Asset & Network & Instability & Linear $R^2$ & Ridge $R^2$ & Network $R^2$ & DM & DM (NW) & DM vs ridge & Regime, divergence', body,
        'Experiments in which the Diebold--Mariano (DM) statistic of the selected dropout model against the linear regression '
        '(Harvey--Leybourne--Newbold correction, squared errors) exceeds 1.96; positive DM values favor the network. '
        '$^{*}$Significant after the Benjamini--Hochberg correction for the false discovery rate (5\\%) within each sample. '
        'DM (NW): with a Newey--West long-run variance (' + str(int(main.nw_lags.iloc[0])) + ' lags). '
        'DM vs ridge: against the ridge regression. Instability: loading instability (Section~\\ref{sec:hypotheses}); '
        'its median across all experiments is ' + f3(main.instability.median()) + '. '
        'Regime, divergence: dropout regime and divergence of the selected model.'))
    return res


def h2_arch(g):
    """H2a for the first-order penalty: median gain over standard dropout in the recurrent networks and in the MLP,
    Wilcoxon p-value of the paired difference, number of pairs."""
    diff = arch_pair(g, g.fo_gain)
    return (g[g.model.isin(RECURRENT)].fo_gain.median(), g[g.model == 'mlp'].fo_gain.median(),
            wilcoxon(diff).pvalue, len(diff))


def robust_stats(sub, seeds, cv, main=None, blocks=None):
    out = {}
    for study in [FUT, EQ]:
        out.setdefault(study, {})
        if main is not None:                          # each test asset once: mean over its three networks
            a = main[main.study == study].groupby('asset').agg(sel=('sel', 'mean'), none=('none', 'mean'),
                                                                ols=('ols', 'mean'), instability=('instability', 'first'))
            out[study]['asset'] = (len(a), (a.sel > a.none).mean(), (a.sel > a.ols).mean(),
                                   *spearmanr(a.instability, a.sel - a.ols))
            m2 = main[main.study == study].dropna(subset=['standard'])
            out[study]['h2_asset'] = h2_arch(m2)
        out[study]['h2_sub'] = h2_arch(sub[sub.study == study].dropna(subset=['standard']))
        if blocks is not None:                        # first-order winners in the subperiod blocks
            bl = blocks[(blocks.study == study) & (blocks.period != '2010_2025') & (blocks.scale == 'linear')]
            out[study]['h2_sub_share'] = (first_share(bl[bl.model.isin(RECURRENT)])[2], first_share(bl[bl.model == 'mlp'])[2])
        s_ = sub[sub.study == study]
        c = cv[cv.study == study]
        sd = seeds[seeds.study == study]
        piv = sd.pivot_table(index=['model', 'asset', 'seed'], columns='kind', values='r2') if len(sd) else pd.DataFrame()
        out[study].update({
            'sub': (len(s_), (s_.sel > s_.none).mean(), (s_.sel > s_.ols).mean(), *spearmanr(s_.instability, s_.sel - s_.ols)),
            'cv': (len(c), (c.sel > c.none).mean(), (c.sel > c.ols).mean(), *spearmanr(c.instability, c.sel - c.ols)),
            'seed': (len(piv), (piv.dropout > piv.none).mean(), piv.groupby(['model', 'asset']).dropout.std().median(),
                     piv.groupby(['model', 'asset']).none.std().median()),
        })
    return out


def tab_robust(rs, out):
    body = []
    for study in [FUT, EQ]:
        r = rs[study]
        body.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{study}}}}} \\\\")
        n, a, b, rho, p = r['sub']
        hr, hm, hp, _ = r['h2_sub']
        body.append(f"Subperiods & {n} & {pct(a)} & {f3(hr)} / {f3(hm)} ({pval(hp)}) & {pct(b)} & {f3(rho)} ({pval(p)}) & -- \\\\")
        n, a, b, rho, p = r['cv']
        body.append(f"Expanding window & {n} & {pct(a)} & -- & {pct(b)} & {f3(rho)} ({pval(p)}) & -- \\\\")
        n, a, sdd, sdn = r['seed']
        body.append(f"Seeds & {n} & {pct(a)} & -- & -- & -- & {f3(sdd)} / {f3(sdn)} \\\\")
        if 'asset' in r:
            n, a, b, rho, p = r['asset']
            hr, hm, hp, _ = r['h2_asset']
            body.append(f"Asset level & {n} & {pct(a)} & {f3(hr)} / {f3(hm)} ({pval(hp)}) & {pct(b)} & {f3(rho)} ({pval(p)}) & -- \\\\")
    write(os.path.join(out, 'tables', 'tab_robust.tex'), table(
        'Robustness of H1, H2 and H3', 'tab:robust', 'lcccccc',
        r'Exercise & $N$ & Dropout $>$ none (\%) & H2a: recurrent / MLP ($p$) & Network $>$ linear (\%) & '
        r'H3a: $\rho$ ($p$) & sd: dropout / none', body,
        'Subperiods: 2010--2014, 2015--2019 and 2020--2025, each with its own split and full search. Expanding window: five '
        'folds, hyperparameters of the main run, refit on each expanding training block. Seeds: search and refit repeated '
        'with five seeds. Subperiods and seeds use four assets per sample (futures: gold, WTI, corn, live cattle; equity: '
        'gold, energy, high technology, utilities) and three networks. Asset level: main run, averaged over the three '
        'networks of each test asset. $N$: number of experiments (asset level: test assets). Dropout $>$ none, Network $>$ '
        'linear: share of experiments in which the selected dropout model has a higher test $R^2$ than the network without regularization '
        'and than the linear model. H2a (first-order penalty) and H3a: as in Table~\\ref{tab:tests}; in the subperiods, H2a '
        'is paired by test asset and period, and in the expanding window, instability compares the loadings of the training '
        'and test blocks of each fold. sd: median across '
        'asset--network pairs of the standard deviation of test $R^2$ across seeds.'))


def tab_appendix(main, out):
    for study, name, label in [(FUT, 'tab_app_futures', 'tab:app_futures'), (EQ, 'tab_app_equity', 'tab:app_equity')]:
        g = main[main.study == study]
        assets = sorted(g.asset.unique(), key=ASSET_ORDER.index)
        body = []
        for a in assets:
            cells = []
            for arch in ARCH:
                r = g[(g.asset == a) & (g.model == arch)]
                cells.append('--' if r.empty else f"{f3(r.none.iloc[0])} & {f3(r.sel.iloc[0])} & " + f"{r.dm_ols.iloc[0]:.1f}".replace('-', '$-$'))
            ols = g[g.asset == a].ols.iloc[0]
            body.append(f"{asset_label(study, a)} & {f3(ols)} & " + ' & '.join(cells) + ' \\\\')
        write(os.path.join(out, 'tables', f'{name}.tex'), table(
            f'Test $R^2$ by asset: {study.lower()}', label, 'lc' + 'ccc' * 3,
            r'Asset & Linear & \multicolumn{3}{c}{LSTM} & \multicolumn{3}{c}{GRU} & \multicolumn{3}{c}{MLP} \\'
            r' \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11} & & none & dropout & DM & none & dropout & DM & none & dropout & DM', body,
            'Linear: linear factor model; none: network without regularization; dropout: selected dropout model; DM: Diebold--Mariano statistic of the '
            'selected model against the linear regression (positive values favor the network).', wide=True))


def tab_app_altreg(main, out):
    rows = []
    for study in [FUT, EQ]:
        g = main[main.study == study]
        assets = sorted(g.asset.unique(), key=ASSET_ORDER.index)
        rows.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{study}}}}} \\\\")
        for a in assets:
            cells = []
            for arch in ARCH:
                r = g[(g.asset == a) & (g.model == arch)]
                cells.append('-- & -- & --' if r.empty else f"{f3(r.standard.iloc[0])} & {f3(r.wd.iloc[0])} & {f3(r.es.iloc[0])}")
            rows.append(f"{asset_label(study, a)} & {f3(g[g.asset == a].ols.iloc[0])} & " + ' & '.join(cells) + ' \\\\')
    write(os.path.join(out, 'tables', 'tab_app_altreg.tex'), table(
        'Test $R^2$ by asset: standard dropout, weight decay and early stopping', 'tab:app_altreg', 'lc' + 'ccc' * 3,
        r'Asset & Linear & \multicolumn{3}{c}{LSTM} & \multicolumn{3}{c}{GRU} & \multicolumn{3}{c}{MLP} \\'
        r' \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11} & & SD & WD & ES & SD & WD & ES & SD & WD & ES', rows,
        'Linear: linear factor model; SD: standard dropout; WD: weight decay; ES: early stopping. Each regime has one search of 50 configurations '
        '(Table~\\ref{tab:setup}).', wide=True))


SHIFT_ASSETS = {FUT: ['c', 'sm', 'w', 'kw', 's', 'cl', 'gc'],
                EQ: ['ind_hlth', 'ind_nodur', 'ind_utils', 'ind_hitec', 'ind_manuf']}
SHIFT_FACTORS = {FUT: [('avg', 'AVG'), ('mom', 'MOM'), ('mkt_rf', 'Mkt-RF')],
                 EQ: [('mkt_rf', 'Mkt-RF'), ('mom', 'MOM'), ('hml', 'HML')]}
EST_END, TEST_START, TEST_END = '2023-09-18', '2023-09-19', '2025-12-31'   # split of the 2010-2025 experiments


def loading(r, name, which):
    """OLS loading of the input whose name is `name` or starts with `name`_ (leave-one-out commodity factors)."""
    k = next(i for i, c in enumerate(r.inputs) if c == name or c.startswith(name + '_'))
    return r[which][k]


def tab_app_shifts(main, out):
    """Linear factor loadings on the estimation and the test block for the assets discussed under H3."""
    body = []
    for study, panel in [(FUT, 'A'), (EQ, 'B')]:
        facs = SHIFT_FACTORS[study]
        title = study.lower() if study == FUT else 'industry portfolios'
        body.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{Panel {panel}: {title}}}}} \\\\")
        body.append('Asset & Instability & ' + ' & '.join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab in facs) + ' \\\\')
        body.append(' & & ' + ' & '.join('estimation & test' for _ in facs) + ' \\\\')
        body.append('\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}')
        for a in SHIFT_ASSETS[study]:
            r = main[(main.study == study) & (main.asset == a)].iloc[0]
            cells = [f"{f3(loading(r, f, 'b_est'))} & {f3(loading(r, f, 'b_test'))}" for f, _ in facs]
            body.append(f"{asset_label(study, a)} & {f3(r.instability)} & " + ' & '.join(cells) + ' \\\\')
        if panel == 'A':
            body.append('\\midrule')
    head, body = body[0][:-3], body[1:]                 # Panel A title becomes the header row
    write(os.path.join(out, 'tables', 'tab_app_shifts.tex'), table(
        'Factor loadings of the linear model before and during the test period', 'tab:app_shifts', 'lccccccc',
        head, body,
        'Ordinary least squares loadings of the linear factor model (all inputs, as in the benchmark regression) estimated on the estimation '
        'block (January 2010 to September 2023) and on the test block (September 2023 to December 2025). AVG, MOM: '
        'the leave-one-out commodity market and momentum factors; Mkt-RF, MOM, HML: the equity market, momentum '
        'and value factors. Instability: loading instability (Section~\\ref{sec:hypotheses}). The first five futures and the first three portfolios have '
        'the most unstable loadings or significant gains; the others are shown for comparison.', wide=True))


def momentum_legs():
    """Share of grain contracts in the short and long legs of the momentum portfolio, estimation vs test months."""
    d = pd.read_csv(os.path.join(FUT_DIR, 'data', 'futures_data.csv'), index_col='Date', parse_dates=True)
    roots = [c[4:] for c in d.columns if c.startswith('ret_')]
    ret = d[[f'ret_{r}' for r in roots]].set_axis(roots, axis=1)
    sig = np.log1p(ret / 100).groupby(ret.index.to_period('M')).sum(min_count=1).rolling(12, min_periods=12).sum()
    sig.index = sig.index + 1                       # signal of the end of M used in M+1, as in build_futures.py
    grains = SECTORS['grains']
    out = {}
    for tag, (a, b) in {'Est': ('2010-01', EST_END[:7]), 'Test': (TEST_START[:7], TEST_END[:7])}.items():
        short, long_ = [], []
        for m in sig.index[(sig.index >= pd.Period(a, 'M')) & (sig.index <= pd.Period(b, 'M'))]:
            r = sig.loc[m].dropna().sort_values()
            k = len(r) // 3
            short.append(len(set(r.index[:k]) & set(grains)) / len(grains))
            long_.append(len(set(r.index[-k:]) & set(grains)) / len(grains))
        out[tag] = (np.mean(short), np.mean(long_))
    cum = np.log1p(ret.loc[TEST_START:TEST_END] / 100).sum()
    corr = {r: (d[f'ret_{r}'].loc['2010':EST_END].corr(d[f'avg_{r}'].loc['2010':EST_END]),
                d[f'ret_{r}'].loc[TEST_START:TEST_END].corr(d[f'avg_{r}'].loc[TEST_START:TEST_END])) for r in ['C', 'SM']}
    return out, cum, corr, grains


def equity_corr():
    ind = pd.read_csv(os.path.join(EQ_DIR, 'data', 'industry_factors.csv'), parse_dates=['Date']).set_index('Date')
    out = {'cum': {c: float(np.log1p((ind[c] - ind.rf).loc[TEST_START:TEST_END] / 100).sum()) for c in
                   ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth', 'Utils', 'Other']}}
    out['cum']['Mkt'] = float(np.log1p(ind.mkt_rf.loc[TEST_START:TEST_END] / 100).sum())
    for c in ['NoDur', 'Hlth', 'Utils', 'HiTec']:
        x = ind[c] - ind.rf
        out[c] = {f: (x.loc['2010':EST_END].corr(ind[f].loc['2010':EST_END]), x.loc[TEST_START:TEST_END].corr(ind[f].loc[TEST_START:TEST_END]))
                  for f in ['mkt_rf', 'mom']}
    return out


EXCHANGE = {'CL': 'NYMEX', 'LCO': 'ICE Europe', 'NG': 'NYMEX', 'HO': 'NYMEX', 'RB': 'NYMEX', 'LGO': 'ICE Europe',
            'GC': 'COMEX', 'SI': 'COMEX', 'HG': 'COMEX', 'PL': 'NYMEX', 'PA': 'NYMEX',
            'C': 'CBOT', 'W': 'CBOT', 'S': 'CBOT', 'BO': 'CBOT', 'SM': 'CBOT', 'KW': 'CBOT', 'O': 'CBOT',
            'KC': 'ICE US', 'SB': 'ICE US', 'CC': 'ICE US', 'CT': 'ICE US', 'OJ': 'ICE US',
            'LC': 'CME', 'LH': 'CME', 'FC': 'CME'}
AUDIT = {}


def futures_audit():
    """Rebuilds every root with data/build_futures.py (nothing is written) and records the construction checks."""
    import contextlib
    import io
    sys.path.insert(0, os.path.join(FUT_DIR, 'data'))
    import build_futures as bf                                   # needs the local LSEG downloads
    stored = pd.read_csv(os.path.join(FUT_DIR, 'data', 'futures_data.csv'), index_col='Date', parse_dates=True)
    ff = bf.read_french(os.path.join(bf.RAW, 'F-F_Research_Data_5_Factors_2x3_daily.csv'), ',Mkt-RF')
    cal = ff.index[(ff.index >= bf.WARMUP) & (ff.index <= bf.END)]
    for root in bf.ROOTS:
        with contextlib.redirect_stdout(io.StringIO()):
            if root in bf.CONTRACT_ROOTS:
                panel, (checked, bad) = bf.panel_from_contracts(root), (0, 0)
            else:
                panel, (checked, bad) = bf.panel_from_continuous(root)
            bf.FALLBACK_DAYS.clear()
            ret, held, carry = bf.root_series(panel, cal)
        r = ret.loc['2010':]
        carry = pd.to_numeric(carry, errors='coerce').reindex(pd.period_range('2010-01', '2025-12', freq='M'))
        AUDIT[root] = dict(carry_cov=float(carry.notna().mean()), carry_n=carry.notna(),
                           slots=len(bf.slots(root)) if root not in bf.CONTRACT_ROOTS else 0, contracts=panel.shape[1], checked=checked, mismatched=bad, continuous=root not in bf.CONTRACT_ROOTS,
                           fallback_2010=sum(t >= pd.Timestamp('2010-01-01') for t in bf.FALLBACK_DAYS),
                           carry=float(carry.mean()) if carry.notna().any() else np.nan,
                           max_diff=float((r - stored[f'ret_{root}'].loc['2010':]).abs().max()))
    return AUDIT


def tab_app_commodities(out):
    d = pd.read_csv(os.path.join(FUT_DIR, 'data', 'futures_data.csv'), index_col='Date', parse_dates=True).loc['2010':]
    body = []
    for sector, roots in SECTORS.items():
        body.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{sector.capitalize()}}}}} \\\\")
        for root in roots:
            a, x = AUDIT[root], d[f'ret_{root}']
            source = f"c1--c{a['slots']}" if a['continuous'] else 'contracts'
            body.append(f"{COMMODITIES[root]} & {EXCHANGE[root]} & {source} & {a['contracts']} & {f3(x.mean() * 252 / 100)} & "
                        f"{f3(x.std() * np.sqrt(252) / 100)} & {f3(a['carry'] / 100)} \\\\")
    write(os.path.join(out, 'tables', 'tab_app_commodities.tex'), table(
        'Commodity futures: data and descriptive statistics, January 2010 to December 2025', 'tab:app_commodities', 'lllcccc',
        'Commodity & Exchange & Data & Contracts & Mean & Volatility & Carry', body,
        'Data: continuous RICs of the first three or four contracts (c1--c3, c1--c4) or individual contracts. Contracts: '
        'number of contract expiries in the price panel, 2008--2026. Mean and volatility: annualized mean and standard '
        'deviation of the daily excess return of the held contract (decimal). Carry: mean annualized log difference between '
        'the prices of the held and the next contract at month ends (decimal); positive values indicate backwardation.',
        wide=True))


def failed_trials():
    """Configurations discarded because the training loss became non-finite, over every grid search of both studies."""
    rows = []
    for study in [FUT, EQ]:
        for root in ['results', 'results_standard', 'results_logalpha', 'results_altreg']:
            for d in finished(os.path.join(STUDIES[study], root)):
                b = pd.read_csv(os.path.join(d, 'best_per_combo.csv'))
                rows.append(b[['Model', 'Divergence', 'N_Failed_Trials']])
    b = pd.concat(rows, ignore_index=True)
    f = b[b.N_Failed_Trials > 0]
    return int(b.N_Failed_Trials.sum()), 50 * len(b), sorted(f.Model.unique()), sorted(f.Divergence.unique())


FIRST_ORDER_ORDER = ['tv', 'wasserstein', 'hellinger']
DIV_LABEL = {'kl': 'Kullback--Leibler', 'reverse_kl': 'Reverse Kullback--Leibler', 'js': 'Jensen--Shannon',
             'pearson_chi_square': 'Pearson $\\chi^2$', 'renyi': 'R\\\'enyi (order 2)', 'jensen_renyi': 'Jensen--R\\\'enyi',
             'bhattacharyya': 'Bhattacharyya', 'tsallis': 'Tsallis (order 2)', 'bregman_euclidean': 'Bregman, Euclidean',
             'bregman_exponential': 'Bregman, exponential', 'bregman_log': 'Bregman, logarithmic',
             'tv': 'Total variation', 'wasserstein': 'Wasserstein-1', 'hellinger': 'Hellinger distance'}
DIVSCALE = {}


def tab_app_divscale(blocks, out):
    res = subprocess.run([sys.executable, os.path.join(HERE, 'divergence_scale.py')], capture_output=True, text=True, check=True)
    DIVSCALE.update(json.loads(res.stdout.strip().splitlines()[-1]))
    lin = blocks[blocks.scale == 'linear']
    body = []
    for order, divs in [('second', [d for d in DIVERGENCES if d not in FIRST_ORDER]), ('first', FIRST_ORDER_ORDER)]:
        body.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{order.capitalize()} order (Proposition~\\ref{{prop:order}})}}}} \\\\")
        for dv in divs:
            v = DIVSCALE[dv]
            wins = [int(((lin.study == st) & (lin.win == dv)).sum()) for st in [FUT, EQ]]
            body.append(f"{DIV_LABEL[dv]} & {sci(v['d_p01'])} & {sci(v['d_p03'])} & {sci(v['grad_p01'])} & {v['exponent']:.2f} & "
                        f"{wins[0]} & {wins[1]} \\\\")
    nb = [int((lin.study == st).sum()) for st in [FUT, EQ]]
    write(os.path.join(out, 'tables', 'tab_app_divscale.tex'), table(
        'Size, gradient and local order of the consistency penalties', 'tab:app_divscale', 'lcccccc',
        r' & \multicolumn{2}{c}{Mean penalty} & Gradient norm & Exponent & \multicolumn{2}{c}{Blocks won} \\'
        r' \cmidrule(lr){2-3}\cmidrule(lr){6-7} Divergence & $p=0.1$ & $p=0.3$ & $p=0.1$ & in $p$ & Futures & Equity', body,
        'Simulation with the implementations used in estimation: final hidden state $\\mathbf h=\\tanh(\\mathbf z)$, '
        '$\\mathbf z\\sim N(\\mathbf 0, I_{256})$, dropout mask with rate $p$, penalty $D(\\operatorname{softmax}(\\mathbf h),'
        '\\operatorname{softmax}(\\tilde{\\mathbf h}))$; means over 2{,}000 draws. Gradient norm: mean Euclidean norm of '
        '$\\partial D/\\partial\\mathbf h$. Exponent: slope of $\\log$ mean penalty on $\\log p$ for $p$ between 0.01 and 0.1 '
        '(Section~\\ref{sec:penalty}). Blocks won: number of regime blocks (test asset, network, penalized regime) in which '
        f'the divergence has the lowest validation error, out of {nb[0]} and {nb[1]}, with the bounded penalty weight; the '
        'chance count is one in 14. Kullback--Leibler and logarithmic Bregman, and Pearson $\\chi^2$ and Tsallis, are '
        'identical penalties (Table~\\ref{tab:divergences}), so their counts differ only through the search.', wide=True))


def compute_record():
    """GPU hours of every grid search and robustness run of both studies (parts of split jobs counted once),
    and the software and hardware recorded by the runs."""
    from datetime import datetime as _dt
    hours, versions, gpus, n = 0.0, set(), set(), 0
    for base in STUDIES.values():
        for f in glob.glob(os.path.join(base, 'results*', '**', 'run_info.json'), recursive=True):
            d = os.path.dirname(f)
            if glob.glob(d + '__part_*'):
                continue                                   # merged folder: its parts are counted instead
            i = json.load(open(f))
            try:
                hours += (_dt.fromisoformat(i['finished']) - _dt.fromisoformat(i['started'])).total_seconds() / 3600
            except (KeyError, ValueError, TypeError):
                continue
            n += 1
            v = i.get('versions', {})
            versions.add((v.get('python'), v.get('torch'), v.get('optuna')))
            if v.get('gpu'):
                gpus.add(v['gpu'])
    return hours, n, versions, gpus


def tab_cost(out):
    path = os.path.join(EQ_DIR, 'results_timing', 'timing.csv')
    if not os.path.isfile(path):
        return
    t = pd.read_csv(path)
    piv = t.groupby(['regime', 'model']).seconds.median().unstack()
    order = ['none', 'standard', 'monte_carlo', 'fixed', 'dynamic', 'adaptive']
    lab = {'none': 'No regularization', 'standard': 'Standard dropout', 'monte_carlo': 'Monte Carlo', 'fixed': 'Fixed + penalty',
           'dynamic': 'Dynamic + penalty', 'adaptive': 'Adaptive + penalty'}
    body = [f"{lab[r]} & " + ' & '.join(f"{piv.loc[r, m]:.2f} ({piv.loc[r, m] / piv.loc['none', m]:.2f})" for m in ARCH) + ' \\\\'
            for r in order if r in piv.index]
    ratio = piv.loc[PENALIZED, ARCH] / piv.loc['none', ARCH]
    COST.update(pen_min=f"{ratio.values.min():.2f}", pen_max=f"{ratio.values.max():.2f}")
    write(os.path.join(out, 'tables', 'tab_app_cost.tex'), table(
        'Computational cost per configuration', 'tab:app_cost', 'lccc',
        'Regime & LSTM & GRU & MLP', body,
        f'Seconds per configuration (20 epochs on the gold training block and evaluation) under identical hyperparameters (3 layers, 256 units, '
        f'$p_0=0.2$, $\\alpha=1$, total variation penalty); medians over {t.rep.nunique()} repetitions in random order '
        'after a warm-up; ratio to no regularization in parentheses.'))


def vif_max(X):
    """Largest variance inflation factor among the columns of X."""
    X = (X - X.mean()) / X.std()
    out = []
    for j in range(X.shape[1]):
        y, Z = X.iloc[:, j].values, np.column_stack([np.ones(len(X)), X.drop(columns=X.columns[j]).values])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        r2 = 1 - np.sum((y - Z @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
        out.append(1 / (1 - r2))
    return max(out)


def tab_vif(out):
    d = pd.read_csv(os.path.join(FUT_DIR, 'data', 'futures_data.csv'), parse_dates=['Date']).set_index('Date')
    macro = ['mkt_rf', 'd_tips', 'd_be', 'dlog_dxy', 'dlog_vix']
    fut = []
    for root in COMMODITIES:
        X = d[[f'avg_{root}', f'carry_{root}', f'mom_{root}'] + macro].copy()
        X['lag1'] = d[f'ret_{root}'].shift(1)
        fut.append(vif_max(X.loc['2010':].dropna()))
    g = pd.read_csv(os.path.join(EQ_DIR, 'data', 'gold_factors.csv'), parse_dates=['Date']).set_index('Date')
    Xg = g[['mkt_rf', 'd_tips', 'd_be', 'dlog_dxy', 'dlog_brent', 'dlog_vix']].copy()
    Xg['lag1'] = (g.gold_ret - g.rf).shift(1)
    ind = pd.read_csv(os.path.join(EQ_DIR, 'data', 'industry_factors.csv'), parse_dates=['Date']).set_index('Date')
    eqcols = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom', 'dlog_vix', 'dlog_dxy', 'd_tips', 'd_be', 'dlog_brent']
    inds = []
    for c in ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth', 'Utils', 'Other']:
        X = ind[eqcols].copy()
        X['lag1'] = (ind[c] - ind.rf).shift(1)
        inds.append(vif_max(X.loc['2010':].dropna()))
    VIF.update({'Fut': np.max(fut), 'Gold': vif_max(Xg.loc['2010':].dropna()), 'Ind': np.max(inds)})
    body = [f"Commodity futures & 9 & 26 & {np.median(fut):.2f} & {np.max(fut):.2f} \\\\",
            f"Gold (GLD) & 7 & 1 & {vif_max(Xg.loc['2010':].dropna()):.2f} & -- \\\\",
            f"Industry portfolios & 12 & 10 & {np.median(inds):.2f} & {np.max(inds):.2f} \\\\"]
    write(os.path.join(out, 'tables', 'tab_app_vif.tex'), table(
        'Multicollinearity of the inputs', 'tab:app_vif', 'lcccc',
        r'Sample & Inputs & Assets & Largest VIF (median) & Largest VIF (max)', body,
        'For each test asset, the largest variance inflation factor among its inputs (factors and lagged own return), '
        'January 2010 to December 2025; median and maximum across assets.'))


def tab_adf(out):
    """Augmented Dickey-Fuller tests of every target and input, 2010-2025 (constant, lag length by AIC)."""
    from statsmodels.tsa.stattools import adfuller
    d = pd.read_csv(os.path.join(FUT_DIR, 'data', 'futures_data.csv'), parse_dates=['Date']).set_index('Date').loc['2010':]
    fut_t = [f'ret_{r}' for r in COMMODITIES]
    fut_i = ['mkt_rf', 'd_tips', 'd_be', 'dlog_dxy', 'dlog_vix'] + [f'{f}_{r}' for f in ['avg', 'carry', 'mom'] for r in COMMODITIES]
    g = pd.read_csv(os.path.join(EQ_DIR, 'data', 'gold_factors.csv'), parse_dates=['Date']).set_index('Date').loc['2010':]
    ind = pd.read_csv(os.path.join(EQ_DIR, 'data', 'industry_factors.csv'), parse_dates=['Date']).set_index('Date').loc['2010':]
    eq_t = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth', 'Utils', 'Other']
    eq_i = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom', 'dlog_vix', 'dlog_dxy', 'd_tips', 'd_be', 'dlog_brent']
    groups = [('Commodity futures', 'returns', d[fut_t]), ('Commodity futures', 'inputs', d[fut_i]),
              ('Gold (GLD)', 'return', g[['gold_ret']]), ('Gold (GLD)', 'inputs', g[['mkt_rf', 'd_tips', 'd_be', 'dlog_dxy', 'dlog_brent', 'dlog_vix']]),
              ('Industry portfolios', 'returns', ind[eq_t]), ('Industry portfolios', 'inputs', ind[eq_i])]
    body = []
    for sample, kind, X in groups:
        stats = [adfuller(X[c].dropna().values, regression='c', autolag='AIC')[:2] for c in X.columns]
        body.append(f"{sample} & {kind} & {X.shape[1]} & {f'{max(s_[0] for s_ in stats):.2f}'.replace('-', '$-$')} & "
                    f"{pval(max(s_[1] for s_ in stats))} \\\\")
    write(os.path.join(out, 'tables', 'tab_app_adf.tex'), table(
        'Unit-root tests', 'tab:app_adf', 'llccc', r'Sample & Series & $N$ & Largest ADF statistic & Largest $p$-value', body,
        'Augmented Dickey--Fuller tests with a constant and the lag length chosen by AIC, January 2010 to December 2025. '
        'The largest statistic and $p$-value over the series of each group are reported; every series rejects a unit root.'))


def fig_divergences(blocks, out):
    cols = [(st, a) for st in [FUT, EQ] for a in ARCH]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
    for ax, scale, title in zip(axes, ['linear', 'log'], [r'Bounded $\alpha$', r'Log-scale $\alpha$']):
        M = np.array([[((blocks.study == st) & (blocks.model == a) & (blocks.scale == scale) & (blocks.win == dv)).sum()
                        for st, a in cols] for dv in DIVERGENCES])
        ax.imshow(M, cmap='Blues', aspect='auto', vmin=0)
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, str(v), ha='center', va='center', fontsize=7, color='white' if v > M.max() * 0.6 else 'black')
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{ARCH_LABEL[a]}\n{'Fut.' if st == FUT else 'Eq.'}" for st, a in cols], fontsize=7)
        ax.axvline(2.5, color='white', linewidth=2)
        ax.axhline(10.5, color='black', linewidth=0.8)
        ax.set_title(title, fontsize=9)
    axes[0].set_yticks(range(len(DIVERGENCES)))
    names = {'kl': 'Kullback–Leibler', 'reverse_kl': 'Reverse Kullback–Leibler', 'js': 'Jensen–Shannon',
             'pearson_chi_square': 'Pearson χ²', 'renyi': 'Rényi (order 2)', 'jensen_renyi': 'Jensen–Rényi',
             'bhattacharyya': 'Bhattacharyya', 'tsallis': 'Tsallis (order 2)', 'bregman_euclidean': 'Bregman, Euclidean',
             'bregman_exponential': 'Bregman, exponential', 'bregman_log': 'Bregman, logarithmic',
             'tv': 'Total variation', 'wasserstein': 'Wasserstein-1', 'hellinger': 'Hellinger distance'}
    axes[0].set_yticklabels([names[dv] for dv in DIVERGENCES], fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'figures', 'fig_app_divergences.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------------------- figures
def fig_h1(main, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=False)
    for ax, study in zip(axes, [FUT, EQ]):
        g = main[main.study == study]
        data, pos, cols = [], [], []
        for i, arch in enumerate(ARCH):
            a = g[g.model == arch]
            for j, (col, c) in enumerate([('ols', '#999999'), ('none', '#E69F00'), ('sel', '#0072B2')]):
                data.append(a[col].values)
                pos.append(i * 4 + j)
                cols.append(c)
        bp = ax.boxplot(data, positions=pos, widths=0.75, patch_artist=True, showfliers=True,
                        flierprops=dict(markersize=3, alpha=0.6), medianprops=dict(color='black', linewidth=1))
        for patch, c in zip(bp['boxes'], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)
            patch.set_linewidth(0.6)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_xticks([1, 5, 9])
        ax.set_xticklabels([ARCH_LABEL[a] for a in ARCH])
        ax.set_title(study, fontsize=10)
        ax.set_ylabel('Test $R^2$')
        ax.grid(axis='y', linewidth=0.3, alpha=0.5)
        ax.spines[['top', 'right']].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.8) for c in ['#999999', '#E69F00', '#0072B2']]
    fig.legend(handles, ['Linear factor model', 'Network, no regularization', 'Network, selected dropout'],
               loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(out, 'figures', 'fig_h1.pdf'))
    plt.close(fig)


REG_STYLE = [('wd', 'Weight decay', '#009E73', 'o'), ('es', 'Early stopping', '#CC79A7', '^'),
             ('fo_r2', 'First-order penalty', '#0072B2', 's')]


def fig_h2(main, blocks, out):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for ax, study, tag in [(axes[0, 0], FUT, '(a)'), (axes[0, 1], EQ, '(b)')]:
        g = main[main.study == study].dropna(subset=['standard', 'wd', 'es', 'fo_r2'])
        for k, (col, lab, c, mk) in enumerate(REG_STYLE):
            data = [(g[g.model == a][col] - g[g.model == a].standard).values for a in ARCH]
            pos = [4 * i_ + k for i_ in range(len(ARCH))]
            bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=False,
                            medianprops=dict(color='black', linewidth=1))
            for patch in bp['boxes']:
                patch.set_facecolor(c)
                patch.set_alpha(0.75)
                patch.set_linewidth(0.6)
            for x_, d_ in zip(pos, data):
                ax.scatter(np.full(len(d_), x_) + np.random.default_rng(0).uniform(-0.25, 0.25, len(d_)), d_, s=6,
                           color=c, alpha=0.5, linewidths=0, marker=mk)
            ax.plot([], [], color=c, linewidth=6, alpha=0.75, label=lab)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_xticks([4 * i_ + 1 for i_ in range(len(ARCH))])
        ax.set_xticklabels([f"{ARCH_LABEL[a]}\n(shortfall {g[g.model == a].short_val.median():.3f})".replace('-', '\u2212') for a in ARCH], fontsize=7)
        ax.set_ylabel('Gain over standard dropout (test $R^2$)', fontsize=7)
        ax.set_title(f"{tag} Gain by network: {'commodity futures' if study == FUT else 'gold and industry portfolios'}", fontsize=8)
        ax.legend(frameon=True, fontsize=6, loc='upper right', framealpha=0.9, edgecolor='none')
        ax.grid(axis='y', linewidth=0.3, alpha=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7)
    for ax, study, tag in [(axes[1, 0], FUT, '(c)'), (axes[1, 1], EQ, '(d)')]:
        x = np.arange(len(ARCH))
        for k, (scale, hatch, alpha) in enumerate([('linear', None, 0.9), ('log', '//', 0.45)]):
            vals = [first_share(blocks[(blocks.study == study) & (blocks.model == a) & (blocks.scale == scale)])[2] for a in ARCH]
            ax.bar(x + (k - 0.5) * 0.38, [100 * v for v in vals], width=0.36, color=COLOR[study], alpha=alpha,
                   hatch=hatch, edgecolor='white', linewidth=0.8,
                   label='bounded $\\alpha$' if scale == 'linear' else 'log-scale $\\alpha$')
        ax.axhline(100 * CHANCE, color='black', linestyle='--', linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_LABEL[a] for a in ARCH], fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Share of blocks (%)', fontsize=7)
        ax.set_title(f"{tag} First-order winners: {'commodity futures' if study == FUT else 'gold and industry portfolios'}", fontsize=8)
        ax.grid(axis='y', linewidth=0.3, alpha=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=6, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'figures', 'fig_h2.pdf'))
    plt.close(fig)


def scatter_gain(main, xcol, xlabel, path, size=(6.2, 4.2)):
    fig, ax = plt.subplots(figsize=size)
    for study, marker in [(FUT, 'o'), (EQ, 's')]:
        g = main[main.study == study]
        sig = (g.dm_ols.abs() > 1.96)
        rho = spearmanr(g[xcol], g.sel - g.ols)[0]
        ax.scatter(g[xcol][~sig], (g.sel - g.ols)[~sig], s=22, marker=marker, facecolors='none', edgecolors=COLOR[study],
                   linewidths=0.9, label=f'{study} ($\\rho$ = {rho:.2f})'.replace('-', '\u2212'))
        ax.scatter(g[xcol][sig], (g.sel - g.ols)[sig], s=22, marker=marker, color=COLOR[study], linewidths=0)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Network minus linear model (test $R^2$)')
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.98, 0.04, 'filled: |DM| > 1.96', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_h3(main, out):
    scatter_gain(main, 'instability', 'Instability of the linear factor loadings (estimation to test block)',
                 os.path.join(out, 'figures', 'fig_h3.pdf'))
    scatter_gain(main, 'ols', 'Test $R^2$ of the linear factor model', os.path.join(out, 'figures', 'fig_app_r2.pdf'))


# ---------------------------------------------------------------------------- numbers
def numbers(main, blocks, logsel, h3, rs, out):
    fut, eq = main[main.study == FUT], main[main.study == EQ]
    m = {}

    def put(name, value):
        m[name] = value

    put('NCommodities', str(len(COMMODITIES)))
    put('NExpFut', str(len(fut)))
    put('NExpEq', str(len(eq)))
    put('NExpAll', str(len(main)))
    d = pd.read_csv(os.path.join(FUT_DIR, 'data', 'futures_data.csv'), parse_dates=['Date'])
    put('NDaysFut', f"{(d.Date >= '2010-01-01').sum():,}".replace(',', '{,}'))
    put('ExpiryChecks', f"{sum(a['checked'] for a in AUDIT.values()):,}".replace(',', '{,}'))
    put('ExpiryMismatches', str(sum(a['mismatched'] for a in AUDIT.values())))
    put('NContinuousRoots', str(sum(a['continuous'] for a in AUDIT.values())))
    put('FallbackDays', str(sum(a['fallback_2010'] for a in AUDIT.values())))
    ncarry = pd.concat([a['carry_n'] for a in AUDIT.values()], axis=1).sum(axis=1)
    put('CarryNMin', str(int(ncarry.min())))              # commodities in the carry sort per month, 2010-2025
    put('NFourSlotRoots', str(sum(a['slots'] == 4 for a in AUDIT.values())))
    put('RebuildMaxDiff', f"{max(a['max_diff'] for a in AUDIT.values()):.1e}")
    sys.path.insert(0, os.path.join(FUT_DIR, 'data'))
    from validate_returns import validate                       # needs the local LSEG downloads
    v = validate()
    put('LSEGChecks', f"{v['n']:,}".replace(',', '{,}'))
    put('LSEGCheckFirst', str(v['first'].year))
    put('LSEGDeviations', str(len(v['deviations'])))
    assert len(v['deviations']) == 2, v['deviations']           # Section 3.1 says "two exceptions"
    for tag, g in [('Fut', fut), ('Eq', eq)]:
        put(f'TrainNone{tag}', f3(g.none_train.median()))
        put(f'TestNone{tag}', f3(g.none.median()))
        put(f'BeatsNone{tag}', pct((g.sel > g.none).mean()))
        put(f'BeatsNoneP{tag}', pval(wilcox_p(g.sel, g.none)))
        put(f'BeatsOLS{tag}', pct((g.sel > g.ols).mean()))
        put(f'SigPlus{tag}', str((g.dm_ols > 1.96).sum()))
        put(f'SigMinus{tag}', str((g.dm_ols < -1.96).sum()))
        s = g.dropna(subset=['standard'])
        put(f'GuidedBeatsStd{tag}', pct((s.sel > s.standard).mean()))
        put(f'GuidedGain{tag}', f3((s.sel - s.standard).median()))
        put(f'GuidedP{tag}', pval(wilcox_p(s.sel, s.standard)))
        study = FUT if tag == 'Fut' else EQ
        b = blocks[blocks.study == study]
        put(f'FirstOrder{tag}', pct(first_share(b[b.scale == 'linear'])[2]))
        put(f'FirstOrderLog{tag}', pct(first_share(b[b.scale == 'log'])[2]) if (b.scale == 'log').any() else '--')
        rec = [first_share(b[(b.scale == 'linear') & (b.model == a)])[2] for a in ['gru', 'lstm']]
        put(f'FirstOrderRec{tag}', f"{pct(min(rec))}--{pct(max(rec))}")
        put(f'AlphaMedLog{tag}', f"{logsel[(logsel.study == study) & logsel.penalized_log].alpha_log.median():.0f}" if len(logsel) else '--')
        put(f'AlphaMedLin{tag}', f"{main[(main.study == study) & main.penalized].alpha.median():.1f}")
        mm = g.merge(logsel[logsel.study == study], on=['study', 'model', 'asset', 'period'])
        put(f'LogBetter{tag}', pct((mm.sel_log > mm.sel).mean()) if len(mm) else '--')
        put(f'LogBetterP{tag}', pval(wilcox_p(mm.sel_log, mm.sel)) if len(mm) else '--')
        rho, p, _ = h3[study]
        put(f'InstabRho{tag}', f3(rho))
        put(f'InstabRhoP{tag}', pval(p))
        g2 = main[main.study == study]
        r_, p_ = spearmanr(g2.ols, g2.sel - g2.ols)
        put(f'Spearman{tag}', f3(r_))
        put(f'SpearmanP{tag}', pval(p_))
        r_, p_ = spearmanr(g2.cond_gain, g2.sel - g2.ols)
        put(f'CondRho{tag}', f3(r_))
        put(f'CondRhoP{tag}', pval(p_))
        put(f'InstabOLSRho{tag}', f3(spearmanr(g2.instability, g2.ols)[0]))
    for tag, study in [('Fut', FUT), ('Eq', EQ)]:
        put(f'SubBeatsNone{tag}', pct(rs[study]['sub'][1]))
        put(f'SubBeatsOLS{tag}', pct(rs[study]['sub'][2]))
        put(f'CVBeatsOLS{tag}', pct(rs[study]['cv'][2]))
        put(f'SubRho{tag}', f3(rs[study]['sub'][3]))
        put(f'SubRhoP{tag}', pval(rs[study]['sub'][4]))
        put(f'CVRhoP{tag}', pval(rs[study]['cv'][4]))
        put(f'CVBeatsNone{tag}', pct(rs[study]['cv'][1]))
        put(f'CVRho{tag}', f3(rs[study]['cv'][3]))
        put(f'SeedBeatsNone{tag}', pct(rs[study]['seed'][1]))
        put(f'SeedSdDrop{tag}', f3(rs[study]['seed'][2]))
        put(f'SeedSdNone{tag}', f3(rs[study]['seed'][3]))
    ts = test_summary(main, blocks, logsel)
    for tag, study in [('Fut', FUT), ('Eq', EQ)]:      # H1: median share of the test-block gap to the linear model closed
        for col, name in [('sel', 'Sel'), ('wd', 'WD'), ('es', 'ES')]:
            put(f'GapClosed{name}{tag}', pct(gap_closed(main[main.study == study], col)))
    for tag, study in [('Fut', FUT), ('Eq', EQ)]:
        put(f'NoneBelowOLS{tag}', ts[study]['h1a'][0].replace('\\%', ''))
        put(f'NoneBelowOLSP{tag}', ts[study]['h1a'][1])
        put(f'McNemarP{tag}', ts[study]['h2c'][1])
        put(f'PenGainRho{tag}', ts[study]['h2a'][0])
        put(f'PenGainRhoP{tag}', ts[study]['h2a'][1])
        put(f'PenGainGapRho{tag}', ts[study]['h2a_gap'][0])
        put(f'PenGainGapRhoP{tag}', ts[study]['h2a_gap'][1])
        put(f'FirstShareRho{tag}', ts[study]['h2c_rho'][0])
        put(f'FirstShareRhoP{tag}', ts[study]['h2c_rho'][1])
        for rg in ['monte_carlo', 'fixed', 'dynamic', 'adaptive']:
            put(f"Sel{rg.replace('_', '').capitalize()}{tag}", str(int((main[main.study == study].regime == rg).sum())))
        put(f'FriedmanP{tag}', FRIEDMAN.get(study, '--'))
    put('VIFMaxFut', f"{VIF.get('Fut', np.nan):.1f}")
    put('VIFMaxGold', f"{VIF.get('Gold', np.nan):.1f}")
    put('VIFMaxInd', f"{VIF.get('Ind', np.nan):.1f}")
    lin = blocks[blocks.scale == 'linear']
    put('WassersteinWins', str(int((lin.win == 'wasserstein').sum())))
    put('NBlocks', str(len(lin)))
    for tag, study in [('Fut', FUT), ('Eq', EQ)]:
        g3 = main[(main.study == study)].dropna(subset=['standard'])
        hi = g3[g3.shortfall > g3.shortfall.median()]
        put(f'FoGainRho{tag}', f3(spearmanr(g3.shortfall, g3.fo_gain)[0]))
        put(f'FoGainRhoP{tag}', pval(spearmanr(g3.shortfall, g3.fo_gain)[1]))
        put(f'FoGainHigh{tag}', f3(hi.fo_gain.median()))
        put(f'FoGainHighP{tag}', pval(wilcoxon(hi.fo_gain).pvalue))
        put(f'EqShare{tag}', pct(g3.eq_share.mean()))
        put(f'EqGain{tag}', f3(g3.eq_gain.median()))
        put(f'EqGainP{tag}', pval(wilcoxon(g3.eq_gain).pvalue))
        put(f'ExAnteRho{tag}', f3(spearmanr(g3.ex_ante, g3.sel - g3.ols)[0]))
        put(f'ExAnteRhoP{tag}', pval(spearmanr(g3.ex_ante, g3.sel - g3.ols)[1]))
        put(f'SigPlusBH{tag}', str(int((main[main.study == study].bh & (main[main.study == study].dm_ols > 0)).sum())))
    for tag, study in [('Fut', FUT), ('Eq', EQ)]:
        g4 = main[main.study == study].dropna(subset=['standard', 'wd', 'es'])
        hi = g4[g4.shortfall > g4.shortfall.median()]
        rec = g4[g4.model.isin(['lstm', 'gru'])]
        for k, K in [('wd', 'WD'), ('es', 'ES')]:
            gain = g4[k] - g4.standard
            put(f'{K}Med{tag}', f3(g4[k].median()))
            put(f'{K}MedRec{tag}', f3(rec[k].median()))
            put(f'{K}BeatsStd{tag}', pct((gain > 0).mean()))
            put(f'{K}StdGain{tag}', f3(gain.median()))
            put(f'{K}StdP{tag}', pval(wilcoxon(gain).pvalue))
            put(f'{K}GainRho{tag}', f3(spearmanr(g4.shortfall, gain)[0]))
            put(f'{K}GainRhoP{tag}', pval(spearmanr(g4.shortfall, gain)[1]))
            put(f'{K}High{tag}', f3((hi[k] - hi.standard).median()))
            put(f'{K}HighShare{tag}', pct((hi[k] > hi.standard).mean()))
            lo = g4[g4.shortfall <= g4.shortfall.median()]
            put(f'{K}Low{tag}', f3((lo[k] - lo.standard).median()))
            put(f'{K}HighP{tag}', pval(wilcoxon(hi[k] - hi.standard).pvalue))
            put(f'SelBeats{K}{tag}', pct((g4.sel > g4[k]).mean()))
            put(f'Sel{K}P{tag}', pval(wilcox_p(g4.sel, g4[k])))
            put(f'Sel{K}High{tag}', f3((hi.sel - hi[k]).median()))
            put(f'FO{K}High{tag}', f3((hi.fo_r2 - hi[k]).median()))
            put(f'{K}BeatsNone{tag}', pct((g4[k] > g4.none).mean()))
            put(f'{K}BeatsOLS{tag}', pct((g4[k] > g4.ols).mean()))
        put(f'StdMed{tag}', f3(g4.standard.median()))
        put(f'OLSMed{tag}', f3(g4.ols.median()))
        put(f'StdMedRec{tag}', f3(rec.standard.median()))
        put(f'SelMedRec{tag}', f3(rec.sel.median()))
        put(f'SelMed{tag}', f3(g4.sel.median()))
        put(f'WDLambdaMed{tag}', f"{g4.wd_lambda.median():.4f}")
        put(f'ESEpochMed{tag}', f"{g4.es_epoch.median():.0f}")
        put(f'ESEpochLeTwo{tag}', pct((g4.es_epoch <= 2).mean()))
        vw = np.array(['std', 'wd', 'es'])[g4[['std_val', 'wd_val', 'es_val']].values.argmin(1)]
        put(f'ESValWins{tag}', str(int((vw == 'es').sum())))
        put(f'FriedmanAltP{tag}', FRIEDMAN.get(study + ' alt', '--'))
    legs, cum, corr, grains = momentum_legs()
    put('GrainShortEst', pct(legs['Est'][0]))
    put('GrainShortTest', pct(legs['Test'][0]))
    put('GrainLongEst', pct(legs['Est'][1]))
    put('GrainLongTest', pct(legs['Test'][1]))
    put('GrainCumMin', f"{cum[grains].min():.2f}".replace('-', '$-$'))
    put('GrainCumMax', f"{cum[grains].max():.2f}".replace('-', '$-$'))
    for r, nm in [('KC', 'Coffee'), ('CC', 'Cocoa'), ('GC', 'Gold'), ('SI', 'Silver')]:
        put(f'Cum{nm}', f"{cum[r]:.2f}")
    put('CornAvgCorrEst', f3(corr['C'][0]))
    put('CornAvgCorrTest', f3(corr['C'][1]))
    put('MealAvgCorrEst', f3(corr['SM'][0]))
    put('MealAvgCorrTest', f3(corr['SM'][1]))
    ec = equity_corr()
    put('CumHiTec', f"{ec['cum']['HiTec']:.2f}")
    put('CumMkt', f"{ec['cum']['Mkt']:.2f}")
    put('HiTecTopCum', 'yes' if max((v, k) for k, v in ec['cum'].items() if k != 'Mkt')[1] == 'HiTec' else 'no')
    for c in ['NoDur', 'Hlth', 'Utils', 'HiTec']:
        for f, F in [('mkt_rf', 'Mkt'), ('mom', 'Mom')]:
            put(f'{c}{F}CorrEst', f3(ec[c][f][0]))
            put(f'{c}{F}CorrTest', f3(ec[c][f][1]))
    for study, tag in [(FUT, 'Fut'), (EQ, 'Eq')]:
        for a in SHIFT_ASSETS[study]:
            r = main[(main.study == study) & (main.asset == a)].iloc[0]
            nm = a.replace('ind_', '').capitalize()
            for f, _ in SHIFT_FACTORS[study]:
                F = {'avg': 'Avg', 'mom': 'Mom', 'mkt_rf': 'Mkt', 'hml': 'Hml'}[f]
                put(f'B{nm}{F}Est', f3(loading(r, f, 'b_est')))
                put(f'B{nm}{F}Test', f3(loading(r, f, 'b_test')))
    for study, assets, f, key in [(FUT, ['c', 'sm', 'w', 'kw', 's'], 'avg', 'GrainAvgDrop'),
                                  (EQ, ['ind_hlth', 'ind_nodur', 'ind_utils'], 'mkt_rf', 'DefMktDrop')]:
        v = [1 - loading(main[(main.study == study) & (main.asset == a)].iloc[0], f, 'b_test') /
             loading(main[(main.study == study) & (main.asset == a)].iloc[0], f, 'b_est') for a in assets]
        put(f'{key}Min', pct(min(v)))
        put(f'{key}Max', pct(max(v)))
    nf, nt, fm, fd = failed_trials()
    put('FailedTrials', str(nf))
    put('TotalTrials', f"{nt:,}".replace(',', '{,}'))
    where = ('feed-forward networks' if fm == ['mlp'] else 'networks') + (' with the Hellinger penalty' if fd == ['hellinger'] else '')
    put('FailedWhere', where)
    for tag, study in [('Fut', FUT), ('Eq', EQ)]:      # where the networks fall behind the linear model (footnote)
        g5 = main[main.study == study]
        put(f'OLSTrainMed{tag}', f3(g5.ols_train.median()))
        put(f'OLSValMed{tag}', f3(g5.ols_val.median()))
        put(f'NetTrainMed{tag}', f3(g5.sel_train.median()))
        put(f'NetValMed{tag}', f3(g5.sel_val.median()))
        put(f'NetBelowOLSTrain{tag}', pct((g5.sel_train < g5.ols_train).mean()))
        a5 = g5.drop_duplicates('asset')
        put(f'OLSWinTest{tag}', f3(a5.olswin_test.median()))
        put(f'OLSWinWorse{tag}', pct((a5.olswin_test < a5.ols).mean()))
        put(f'NetToOLS{tag}', pct((g5[g5.ols > 0.05].sel / g5[g5.ols > 0.05].ols).median()))
    ind = pd.read_csv(os.path.join(EQ_DIR, 'data', 'industry_factors.csv'), parse_dates=['Date']).set_index('Date').loc['2010':EST_END]
    capm = []
    for c in ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth', 'Utils', 'Other']:
        y5 = (ind[c] - ind.rf).values
        A = np.column_stack([np.ones(len(ind)), ind.mkt_rf.values])
        e5 = y5 - A @ np.linalg.lstsq(A, y5, rcond=None)[0]
        capm.append(1 - (e5 ** 2).sum() / ((y5 - y5.mean()) ** 2).sum())
    put('CAPMRsqInd', pct(np.median(capm)))             # market-only R2 of the industries, estimation period
    for tag, study in [('Fut', FUT), ('Eq', EQ)]:      # H2 with overfitting measured before the test block
        g6 = main[main.study == study].dropna(subset=['standard', 'wd', 'es'])
        rec6, mlp6 = g6[g6.model.isin(RECURRENT)], g6[g6.model == 'mlp']
        put(f'ShortValRec{tag}', f3(rec6.short_val.median()))
        put(f'ShortValMLP{tag}', f3(mlp6.short_val.median()))
        put(f'ShortValMed{tag}', f3(g6.short_val.median()))
        put(f'GapTVRec{tag}', f3(rec6.gap_tv.median()))
        put(f'GapTVMLP{tag}', f3(mlp6.gap_tv.median()))
        for K, gain in [('WD', g6.wd - g6.standard), ('ES', g6.es - g6.standard), ('FO', g6.fo_r2 - g6.standard)]:
            put(f'Gain{K}Rec{tag}', f3(gain[g6.model.isin(RECURRENT)].median()))
            put(f'Gain{K}MLP{tag}', f3(gain[g6.model == 'mlp'].median()))
            diff = arch_pair(g6, gain)
            put(f'Gain{K}PairP{tag}', pval(wilcoxon(diff).pvalue))
            put(f'Gain{K}PairShare{tag}', pct((diff > 0).mean()))
            put(f'Gain{K}RecP{tag}', pval(wilcoxon(gain[g6.model.isin(RECURRENT)]).pvalue))
            r_, p_ = spearmanr(g6.short_val, gain)
            put(f'{K}ValRho{tag}', f3(r_))
            put(f'{K}ValRhoP{tag}', pval(p_))
        within = [spearmanr(h.short_val, h[col] - h.standard)          # the same correlations within each network
                  for mdl in ARCH for h in [g6[g6.model == mdl]] for col in ['wd', 'es']]
        put(f'WithinGainRhoMin{tag}', f3(min(r[0] for r in within)))
        put(f'WithinGainRhoMax{tag}', f3(max(r[0] for r in within)))
        put(f'WithinGainRhoMinP{tag}', pval(min(r[1] for r in within)))
        if tag == 'Fut':                                  # the text says that none of the six is significant
            assert min(r[1] for r in within) >= 0.05
        r_, p_ = spearmanr(main[main.study == study].short_val, main[main.study == study].first_share)
        put(f'FirstShareValRho{tag}', f3(r_))
        put(f'FirstShareValRhoP{tag}', pval(p_))
        rec_m = main[(main.study == study) & main.model.isin(RECURRENT)]    # within the recurrent networks
        r_, p_ = spearmanr(rec_m.short_val, rec_m.first_share)
        put(f'FirstShareValRhoRec{tag}', f3(r_))
        put(f'FirstShareValRhoRecP{tag}', pval(p_))
        # the strength selected on the validation block, within each network: weight decay and early stopping
        n_sig, n_sign, n_any, max_abs = 0, 0, 0, 0.0
        for mdl in ARCH:
            h = main[(main.study == study) & (main.model == mdl)].dropna(subset=['wd_lambda', 'es_epoch'])
            for key, col, sign in [('WDLam', np.log(h.wd_lambda), 1), ('ESEp', h.es_epoch, -1)]:
                r_, p_ = spearmanr(h.short_val, col)
                put(f'{key}Rho{ARCH_LABEL[mdl]}{tag}', f3(r_))
                n_sig += (p_ < 0.05) and (np.sign(r_) == sign)
                n_sign += np.sign(r_) == sign
                n_any += p_ < 0.05
                max_abs = max(max_abs, abs(r_))
        put(f'StrengthRhoSig{tag}', str(int(n_sig)))
        put(f'StrengthRhoSign{tag}', str(int(n_sign)))
        put(f'StrengthRhoMaxAbs{tag}', f3(max_abs))
        if tag == 'Fut':                                  # the text says that all six have the expected sign
            assert n_sign == 6, n_sign
        if tag == 'Eq':                                   # the text says that none is significant in the equity sample
            assert n_any == 0, n_any
        g9 = main[main.study == study].dropna(subset=['std_rate'])
        put(f'StdNearBound{tag}', pct((g9.std_rate > 0.45).mean()))
        hr, hm, hp, n2 = rs[study]['h2_sub']
        put(f'SubFOGainRec{tag}', f3(hr))
        put(f'SubFOGainMLP{tag}', f3(hm))
        put(f'SubFOGainP{tag}', pval(hp))
        put(f'SubFOPairs{tag}', str(n2))
        sr, sm = rs[study]['h2_sub_share']
        put(f'SubFirstRec{tag}', pct(sr))
        put(f'SubFirstMLP{tag}', pct(sm))
        a6 = g6.groupby('asset')[['wd', 'es', 'standard']].mean()
        put(f'AssetWDBeatsStd{tag}', pct((a6.wd > a6.standard).mean()))
        put(f'AssetWDStdP{tag}', pval(wilcoxon(a6.wd - a6.standard).pvalue))
        put(f'AssetESBeatsStd{tag}', pct((a6.es > a6.standard).mean()))
        put(f'AssetESStdP{tag}', pval(wilcoxon(a6.es - a6.standard).pvalue))
        put(f'NAssets{tag}', str(len(a6)))
        n_, a_, b_, rho_, p_ = rs[study]['asset']
        put(f'AssetBeatsNone{tag}', pct(a_))
        put(f'AssetBeatsOLS{tag}', pct(b_))
        put(f'AssetInstabRho{tag}', f3(rho_))
        put(f'AssetInstabRhoP{tag}', pval(p_))
    if COST:
        put('CostPenMin', COST['pen_min'])
        put('CostPenMax', COST['pen_max'])
    put('TotalRsqDiff', f3((main.sel_total - main.sel).abs().median()))
    put('TotalRsqDiffMax', f3((main.sel_total - main.sel).abs().max()))
    hours, nruns, versions, gpus = compute_record()
    put('GPUHours', f"{round(hours, -1):,.0f}".replace(',', '{,}'))   # every run in the result folders
    put('NRuns', f"{nruns:,}".replace(',', '{,}'))
    (py, torch_v, optuna_v), = versions if len(versions) == 1 else [sorted(versions)[0]]
    put('PyVer', py)
    put('TorchVer', torch_v.split('+')[0])
    put('OptunaVer', optuna_v)
    put('GPUList', ' and '.join(sorted(g.replace('NVIDIA ', 'NVIDIA~') for g in gpus)))
    sp = json.load(open(glob.glob(os.path.join(FUT_DIR, 'results', 'mlp', '*_2010_2025', 'run_info.json'))[0]))['splits']
    fmt = lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%-d %B %Y')
    for part, P in [('train', 'Train'), ('val', 'Val'), ('test', 'Test')]:
        put(f'{P}StartDate', fmt(sp[part]['start']))
        put(f'{P}EndDate', fmt(sp[part]['end']))
        put(f'N{P}Days', f"{sp[part]['n']:,}".replace(',', '{,}'))
    v = DIVSCALE
    put('ExpWass', f"{v['wasserstein']['exponent']:.2f}")
    put('ExpHel', f"{v['hellinger']['exponent']:.2f}")
    put('ExpTV', f"{v['tv']['exponent']:.2f}")
    sec = [v[d]['exponent'] for d in DIVERGENCES if d not in FIRST_ORDER]
    put('ExpSecMin', f"{min(sec):.2f}")
    put('ExpSecMax', f"{max(sec):.2f}")
    put('RatioWassKL', f"{v['wasserstein']['d_p01'] / v['kl']['d_p01']:.0f}")
    put('RatioHelKL', f"{v['hellinger']['d_p01'] / v['kl']['d_p01']:.1f}")
    put('RatioTVKL', f"{v['tv']['d_p01'] / v['kl']['d_p01']:.1f}")
    for tag, study in [('Fut', FUT), ('Eq', EQ)]:
        for scale, sfx in [('linear', ''), ('log', 'Log')]:
            b7 = blocks[(blocks.study == study) & (blocks.scale == scale)]
            put(f'WinsWass{sfx}{tag}', str(int((b7.win == 'wasserstein').sum())))
            for dv, nm in [('kl', 'KL'), ('bregman_log', 'BLog'), ('pearson_chi_square', 'Chi'), ('tsallis', 'Tsa')]:
                put(f'Wins{nm}{sfx}{tag}', str(int((b7.win == dv).sum())))
            put(f'WinsHel{sfx}{tag}', str(int((b7.win == 'hellinger').sum())))
            put(f'WinsTV{sfx}{tag}', str(int((b7.win == 'tv').sum())))
            put(f'ChanceWins{sfx}{tag}', f"{len(b7) / 14:.0f}")
        g8 = main[main.study == study]
        r_, p_ = spearmanr(g8.gap_tv, g8.first_share)
        put(f'FirstShareGapRho{tag}', f3(r_))
        put(f'FirstShareGapRhoP{tag}', pval(p_))
        put(f'SigPlusNW{tag}', str(int((g8.dm_ols_nw > 1.96).sum())))
        put(f'SigMinusNW{tag}', str(int((g8.dm_ols_nw < -1.96).sum())))
        put(f'SigPlusHolm{tag}', str(int((g8.holm & (g8.dm_ols > 0)).sum())))
        rec8 = g8[g8.model.isin(RECURRENT)]
        put(f'RecBeatOLS{tag}', pct((rec8.sel > rec8.ols).mean()))
        put(f'RecBeatOLSWin{tag}', pct((rec8.sel > rec8.olswin_test).mean()))
        put(f'OLSWinMed{tag}', f3(g8.drop_duplicates('asset').olswin_test.median()))
    put('NWLags', str(int(main.nw_lags.iloc[0])))
    put('InstabRhoAll', f3(h3['all'][0]))
    put('InstabRhoPAll', pval(h3['all'][1]))
    sig = main[main.dm_ols > 1.96]
    put('InstabMedSig', f3(sig.instability.median()))
    put('InstabMedAll', f3(main.instability.median()))
    c = main[(main.study == FUT) & (main.asset == 'c')]
    put('CornCondGain', f3(c.cond_gain.mean()))
    ng = main[(main.study == FUT) & (main.asset == 'ng')]
    put('NGOLS', f3(ng.ols.iloc[0]))
    put('NGInstab', f3(ng.instability.iloc[0]))
    rank = int((main[(main.study == FUT)].drop_duplicates('asset').instability < ng.instability.iloc[0]).sum()) + 1
    put('NGInstabRankText', 'the lowest' if rank == 1 else f'the {rank}{"nd" if rank == 2 else "rd" if rank == 3 else "th"} lowest')
    put('CornDeepGain', f3((c.sel - c.ols).mean()))
    for tag, study in [('Fut', FUT), ('Eq', EQ)]:
        g = main[main.study == study]
        put(f'SigPlusAbs{tag}', str(int((g.dm_ols_abs > 1.96).sum())))
        put(f'SigMinusAbs{tag}', str(int((g.dm_ols_abs < -1.96).sum())))
        dp = duplicate_placebo(study)
        put(f'DupDiffMed{tag}', f3(dp['dup']))
        put(f'DistinctDiffMed{tag}', f3(dp['dist']))
        put(f'DupSameShare{tag}', pct(dp['same']))
        g4 = g.dropna(subset=['standard', 'wd'])
        const = g4[g4.wd_sdratio < 0.01]
        put(f'WDConst{tag}', str(len(const)))
        put(f'WDBeatsStdExcl{tag}', pct((g4[g4.wd_sdratio >= 0.01].wd > g4[g4.wd_sdratio >= 0.01].standard).mean()))
        if tag == 'Fut':
            put('WDConstList', ' and '.join(f"{asset_label(study, r.asset).lower()} with the {ARCH_LABEL[r.model]}"
                                           for _, r in const.sort_values('asset').iterrows()))
    # H3: the source of the gains (ridge regression, tracking share), the Chow test and the rolling regression
    for tag, study in [('Fut', FUT), ('Eq', EQ)]:
        g = main[main.study == study]
        a = g.drop_duplicates('asset')
        put(f'ChowSig{tag}', str(int((a.chow_p < 0.05).sum())))
        put(f'RidgeMed{tag}', f3(a.ridge_test.median()))
        put(f'SelBeatsRidgeSig{tag}', str(int((g.dm_sel_ridge > 1.96).sum())))
        put(f'SelLosesRidgeSig{tag}', str(int((g.dm_sel_ridge < -1.96).sum())))
        r_, p_ = spearmanr(g.ridge_test - g.ols, g.sel - g.ols)
        put(f'RidgeGainRho{tag}', f3(r_))
        put(f'RidgeGainRhoP{tag}', pval(p_))
        sg = g[g.dm_ols > 1.96]
        put(f'SigMatchRidge{tag}', str(int((sg.dm_sel_ridge <= 1.96).sum())))
        put(f'Track{tag}', f3(g.track.median()))
        put(f'TrackRest{tag}', f3(g.track_rest.median()))
        put(f'TrackRestP{tag}', pval(wilcoxon(g.track_rest).pvalue))
        put(f'TrackRestShare{tag}', pct((g.track_rest > 0).mean()))
        put(f'TrackCondRest{tag}', f3(a.track_cond_rest.median()))
        for W, name in [(250, 'Short'), (500, 'Long')]:      # macro names cannot contain digits
            put(f'Roll{name}Med{tag}', f3(a[f'roll{W}_test'].median()))
        put(f'RollExpMed{tag}', f3(a.roll_exp_test.median()))
        put(f'ShortFrozenMed{tag}', f3(a.short_frozen_test.median()))
        for k, K in [('none', 'None'), ('sel', 'Sel'), ('wd', 'WD'), ('es', 'ES')]:   # linearity of the fits
            put(f'Lin{K}{tag}', pct(g[f'lin_{k}'].median()))
        put(f'RollBeatsSel{tag}', str(int((g.dm_roll250_sel > 1.96).sum())))
        put(f'RollLosesSel{tag}', str(int((g.dm_roll250_sel < -1.96).sum())))
        for k, col in [('WD', 'wd'), ('ES', 'es')]:
            r_, p_ = spearmanr(g.instability, g[col] - g.ols)
            put(f'InstabRho{k}{tag}', f3(r_))
            put(f'InstabRhoP{k}{tag}', pval(p_))
    sig = main[main.dm_ols > 1.96]
    put('SigPlusAll', str(len(sig)))
    put('SigMatchRidgeAll', str(int((sig.dm_sel_ridge <= 1.96).sum())))
    exc = sig[sig.dm_sel_ridge > 1.96]
    in_text = {'w': 'the Chicago wheat contract', 'kw': 'the Kansas City wheat contract'}
    put('RidgeExceptions', ' and '.join(f"{in_text.get(r.asset, asset_label(r.study, r.asset))} with the {ARCH_LABEL[r.model]}"
                                        for _, r in exc.iterrows()))
    # the text says that for these contracts validation selects (almost) no shrinkage
    assert (exc.ridge_lambda <= 1e-3).all(), exc[['asset', 'model', 'ridge_lambda']]
    a = main.groupby(['study', 'asset']).agg(instability=('instability', 'first'), gain=('sel', 'mean'), ols=('ols', 'first'))
    r_, p_ = spearmanr(a.instability, a.gain - a.ols)
    put('NAssetsAll', str(len(a)))
    put('PooledInstabRho', f3(r_))
    put('PooledInstabRhoP', pval(p_))
    put('FirstOrderMLP', pct(first_share(blocks[(blocks.scale == 'linear') & (blocks.model == 'mlp')])[2]))
    for tag, study in [('Fut', FUT), ('Eq', EQ)]:
        put(f'FirstOrderMLP{tag}', pct(first_share(blocks[(blocks.scale == 'linear') & (blocks.model == 'mlp')
                                                          & (blocks.study == study)])[2]))
    put('AlphaMedLin', f"{main[main.penalized].alpha.median():.1f}")
    lines = ['% Generated by 3_analysis/make_tables.py on ' + datetime.now().isoformat(timespec='seconds'),
             '% Do not edit by hand.']
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    write(os.path.join(out, 'numbers.tex'), '\n'.join(lines) + '\n')
    return m


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(out, 'figures'), exist_ok=True)

    grids, blocks, logs, prov, seeds, cvs = [], [], [], {}, [], []
    for study in [FUT, EQ]:
        g, b, la, folders = load_grid(study)
        prov[study] = check_code(study, folders)
        grids.append(g)
        blocks.append(b)
        logs.append(la)
        seeds.append(load_seeds(study))
        cvs.append(load_cv(study))
    allg = pd.concat(grids, ignore_index=True)
    blocks = pd.concat(blocks, ignore_index=True)
    logsel = pd.concat(logs, ignore_index=True)
    seeds = pd.concat(seeds, ignore_index=True)
    cv = pd.concat(cvs, ignore_index=True)
    allg['shortfall'] = allg.ols - allg.none          # how far the unregularized network falls below the linear model
    allg['gap'] = allg.none_train - allg.none         # train-test gap of the unregularized network
    allg['pen_gain'] = allg.sel - allg.standard       # gain of the selected model over plain dropout
    fs = (blocks[blocks.scale == 'linear'].assign(f=lambda x: x.win.isin(FIRST_ORDER))
          .groupby(['study', 'model', 'asset', 'period']).f.mean().rename('first_share').reset_index())
    allg = allg.merge(fs, on=['study', 'model', 'asset', 'period'], how='left')
    inst = pd.concat([load_instability(st) for st in [FUT, EQ]], ignore_index=True)
    allg = allg.merge(inst[inst.fold == 0].drop(columns='fold'), on=['study', 'asset', 'period'], how='left')
    cv = cv.merge(inst[inst.period == '2010_2025'][['study', 'asset', 'fold', 'instability']], on=['study', 'asset', 'fold'], how='left')
    # overfitting measured before the test block: linear model minus unregularized network on the validation
    # block, both estimated on the training block (search stage); and the network's training-validation gap
    allg['short_val'] = allg.ols_val - allg.none_val
    allg['gap_tv'] = allg.none_train - allg.none_val
    main_ = allg[allg.period == '2010_2025'].reset_index(drop=True)
    hb = pd.concat([load_h3bench(st) for st in [FUT, EQ]], ignore_index=True)
    n0 = len(main_)
    main_ = main_.merge(hb, on=['study', 'asset', 'model'], how='left', validate='one_to_one')
    assert len(main_) == n0 and main_.ridge_test.notna().all(), 'H3 benchmarks missing for some experiments'
    from statsmodels.stats.multitest import multipletests
    main_['bh'] = False
    for st in [FUT, EQ]:
        m_ = main_.study == st
        main_.loc[m_, 'bh'] = multipletests(main_.loc[m_, 'dm_ols_p'], alpha=0.05, method='fdr_bh')[0]
        main_.loc[m_, 'holm'] = multipletests(main_.loc[m_, 'dm_ols_p'], alpha=0.05, method='holm')[0]
    sub = allg[allg.period != '2010_2025'].reset_index(drop=True)
    mblocks = blocks[blocks.period == '2010_2025']
    mlog = logsel[logsel.period == '2010_2025']

    tab_desc(out)
    tab_divergences(out)
    tab_setup(out)
    tab_regimes(main_, out)
    tab_tests(main_, mblocks, mlog, out)
    h3 = h3_stats(main_)
    rs = robust_stats(sub, seeds, cv, main_, blocks)
    tab_robust(rs, out)
    tab_appendix(main_, out)
    tab_app_altreg(main_, out)
    tab_app_shifts(main_, out)
    tab_app_divscale(mblocks, out)
    futures_audit()
    tab_app_commodities(out)
    tab_cost(out)
    tab_vif(out)
    tab_adf(out)
    plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})
    fig_h1(main_, out)
    fig_h2(main_, mblocks, out)
    fig_h3(main_, out)
    fig_divergences(mblocks, out)
    nums = numbers(main_, mblocks, mlog, h3, rs, out)

    manifest = {'generated': datetime.now().isoformat(timespec='seconds'),
                'experiments': {s: int((main_.study == s).sum()) for s in [FUT, EQ]},
                'subperiod_experiments': {s: int((sub.study == s).sum()) for s in [FUT, EQ]},
                'logalpha_experiments': {s: int((mlog.study == s).sum()) for s in [FUT, EQ]},
                'altreg_experiments': {s: int(main_[main_.study == s].wd.notna().sum()) for s in [FUT, EQ]},
                'seed_runs': {s: int(seeds[seeds.study == s][['model', 'asset', 'seed']].drop_duplicates().shape[0]) for s in [FUT, EQ]},
                'cv_folds': {s: int((cv.study == s).sum()) for s in [FUT, EQ]},
                'code_md5': prov, 'numbers': nums}
    write(os.path.join(out, 'MANIFEST.json'), json.dumps(manifest, indent=1))
    main_.to_csv(os.path.join(out, 'tables', 'experiments.csv'), index=False)
    print(json.dumps({k: manifest[k] for k in ['experiments', 'subperiod_experiments', 'logalpha_experiments', 'altreg_experiments',
                                               'seed_runs', 'cv_folds']}, indent=1))
    print(f"✅ tables, figures and numbers.tex written to {out}")


if __name__ == '__main__':
    main()
