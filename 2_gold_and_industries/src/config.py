"""
config.py
---------
Central configuration: contemporaneous asset-pricing factor models.

    r_{i,t} - rf_t = g( F_t , r_{i,t-1} - rf_{t-1} ) + e_t

The target is the daily excess return of a test asset; the inputs are the factor
realizations of the same day and the asset's own lagged excess return. Recurrent
models see the last WINDOW days of these inputs; the MLP sees day t only. The
asset's own return of day t never enters the inputs (data.assert_target_not_in_inputs).
"""
import random
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Test assets and factors
# ─────────────────────────────────────────────────────────────────────────────

# Gold: literature-based drivers (all same-day changes / returns)
#   mkt_rf     equity market      Baur & Lucey (2010); Baur & McDermott (2010)
#   d_tips     10y real yield     Erb & Harvey (2013); Chicago Fed Letter 464 (2021)
#   d_be       10y breakeven      Chicago Fed (2021); O'Connor et al. (2015)
#   dlog_dxy   US dollar          Pukthuanthong & Roll (2011)
#   dlog_brent oil                Reboredo (2013)
#   dlog_vix   uncertainty        Chicago Fed (2021); safe-haven literature
GOLD_FACTORS = ['mkt_rf', 'd_tips', 'd_be', 'dlog_dxy', 'dlog_brent', 'dlog_vix']

# Industry portfolios: Fama-French five factors plus Carhart momentum, augmented with
# a macro block (VIX, dollar, real rates, breakeven inflation, oil), in the
# spirit of macroeconomic factor models (Chen, Roll & Ross, 1986). Max VIF 2.9.
FF_FACTORS = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']
MACRO_FACTORS = ['dlog_vix', 'dlog_dxy', 'd_tips', 'd_be', 'dlog_brent']
STOCK_FACTORS = FF_FACTORS + MACRO_FACTORS
INDUSTRIES = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth', 'Utils', 'Other']

ASSETS = {'gold': {'file': 'gold_factors.csv', 'return': 'gold_ret', 'factors': GOLD_FACTORS,
                   'name': 'Gold (GLD)'}}
for _p in INDUSTRIES:
    ASSETS[f'ind_{_p.lower()}'] = {'file': 'industry_factors.csv', 'return': _p,
                                   'factors': STOCK_FACTORS, 'name': f'Industry: {_p}'}

# Sample periods: [start, end)
PERIODS = {
    '2010_2015': ('2010-01-01', '2015-01-01'),
    '2015_2020': ('2015-01-01', '2020-01-01'),
    '2020_2025': ('2020-01-01', '2026-01-01'),
    '2010_2025': ('2010-01-01', '2026-01-01'),
}

EXPERIMENTS = {
    f'{asset}_{period}': {'asset': asset, 'period': PERIODS[period],
                          'name': f"{ASSETS[asset]['name']} ({period.replace('_', '-')})"}
    for asset in ASSETS for period in PERIODS
}

MODEL_TYPES = ['lstm', 'gru', 'mlp']
WINDOW = 20             # days of inputs seen by the recurrent models

# Chronological split: TimeSeriesSplit(N_SPLITS) -> 7 equal blocks,
# train = blocks 1-5, validation = block 6, test = block 7
N_SPLITS = 6


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_EPOCHS = 20      # Epochs per Optuna trial (fit on train, score on validation)
RETRAIN_EPOCHS = 30     # Epochs for the final refit on train + validation
LEARNING_RATE = 1e-3
BATCH_SIZE = 64          # mini-batch size
GRADIENT_CLIP = 5.0
MC_SAMPLES = 50         # Stochastic forward passes averaged by MC dropout
N_TRIALS = 50


# ─────────────────────────────────────────────────────────────────────────────
# Dropout regimes and divergences
# ─────────────────────────────────────────────────────────────────────────────

DROPOUT_TYPES = ['none', 'monte_carlo', 'standard', 'fixed', 'dynamic', 'adaptive', 'weight_decay', 'early_stopping']

# Alternative regularizers of the network without dropout, used only for comparison:
# an L2 penalty through Adam's weight decay, and early stopping on the validation block
# (the best epoch is restored; the refit uses that number of epochs scaled by
# RETRAIN_EPOCHS / SEARCH_EPOCHS, the same ratio as every other regime).
ALT_REGULARIZERS = ['weight_decay', 'early_stopping']
EARLY_STOPPING_PATIENCE = 5

# Regimes whose loss includes alpha * divergence (and that are paired with
# every divergence). 'none', 'monte_carlo' and 'standard' use plain MSE and run once.
# 'standard' is plain dropout at the tuned rate (Srivastava et al., 2014): it is
# the ablation benchmark for the divergence penalty, not a candidate in the selection.
PENALIZED_TYPES = ['fixed', 'dynamic', 'adaptive']
SELECTION_EXCLUDES = ['none', 'standard'] + ALT_REGULARIZERS

DIVERGENCE_METHODS = [
    'js', 'kl', 'reverse_kl', 'hellinger', 'tv', 'pearson_chi_square',
    'wasserstein', 'bregman_euclidean', 'bregman_exponential',
    'bregman_log', 'renyi', 'jensen_renyi', 'bhattacharyya', 'tsallis'
]

MAX_DROPOUT_RATE = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search space (Optuna, TPE sampler)
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    'num_layers': (2, 4),               # int, inclusive
    'hidden_size': [128, 256, 512],     # categorical
    'base_rate': (0.01, 0.50),          # float; all regimes except 'none'
    'alpha': (0.0, 10.0, 0.5),          # float with step; penalized regimes only
    'alpha_log': (1e-3, 1e3),           # log-uniform alternative (--alpha-scale log): lets every
                                        # divergence reach its own effective penalty strength
    'weight_decay': (1e-6, 1e-1),       # log-uniform; weight_decay regime only
}
