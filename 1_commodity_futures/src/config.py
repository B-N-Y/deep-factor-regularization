"""
config.py
---------
Central configuration: contemporaneous factor models of commodity futures.

    r_{i,t} = g( F_t , r_{i,t-1} ) + e_t

The target is the daily excess return of a commodity futures position (a
futures position needs no capital, so its return is already an excess return);
the inputs are the factor realizations of the same day and the asset's own
lagged return. Recurrent
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

# Commodity futures: 26 roots in five sectors (LSEG continuous and individual
# contracts, rolled before the expiry month; see data/build_futures.py).
COMMODITIES = {
    'CL': 'WTI crude oil', 'LCO': 'Brent crude oil', 'NG': 'Natural gas', 'HO': 'Heating oil',
    'RB': 'RBOB gasoline', 'LGO': 'Gasoil',
    'GC': 'Gold', 'SI': 'Silver', 'HG': 'Copper', 'PL': 'Platinum', 'PA': 'Palladium',
    'C': 'Corn', 'W': 'Wheat (CBOT)', 'S': 'Soybeans', 'BO': 'Soybean oil', 'SM': 'Soybean meal',
    'KW': 'Wheat (KC HRW)', 'O': 'Oats',
    'KC': 'Coffee', 'SB': 'Sugar', 'CC': 'Cocoa', 'CT': 'Cotton', 'OJ': 'Orange juice',
    'LC': 'Live cattle', 'LH': 'Lean hogs', 'FC': 'Feeder cattle',
}
SECTORS = {
    'energy': ['CL', 'LCO', 'NG', 'HO', 'RB', 'LGO'], 'metals': ['GC', 'SI', 'HG', 'PL', 'PA'],
    'grains': ['C', 'W', 'S', 'BO', 'SM', 'KW', 'O'], 'softs': ['KC', 'SB', 'CC', 'CT', 'OJ'],
    'livestock': ['LC', 'LH', 'FC'],
}

# Factors (all same-day returns / changes):
#   avg, carry, mom   commodity market, carry (basis) and momentum factors of
#                     Bakshi, Gao & Rossi (2019); built leave-one-out, i.e. from
#                     the other 25 commodities, so the asset never enters its own factors
#   mkt_rf            equity market excess return (Fama-French)
#   d_tips, d_be      10-year real yield and breakeven inflation (FRED)
#   dlog_dxy          US dollar index
#   dlog_vix          uncertainty (VIX)
COMMODITY_FACTORS = ['avg', 'carry', 'mom']
MACRO_FACTORS = ['mkt_rf', 'd_tips', 'd_be', 'dlog_dxy', 'dlog_vix']

ASSETS = {
    root.lower(): {'file': 'futures_data.csv', 'return': f'ret_{root}', 'rf': None,
                   'factors': [f'{f}_{root}' for f in COMMODITY_FACTORS] + MACRO_FACTORS,
                   'name': name}
    for root, name in COMMODITIES.items()
}

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
