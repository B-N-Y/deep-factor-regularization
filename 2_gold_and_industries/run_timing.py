#!/usr/bin/env python
"""
run_timing.py
-------------
Controlled computational-cost measurement: every architecture x dropout regime is
trained with the SAME hyperparameters (so differences reflect the regime, not the
model size Optuna happened to choose) for SEARCH_EPOCHS epochs on the gold training
block, repeated --reps times. Writes results_timing/timing.csv.

GPU nodes are shared, so the load can drift during the run. Each architecture
first gets one untimed warm-up fit (CUDA context, cuDNN and allocator start-up),
and every repetition then measures all architecture x regime pairs in a new
random order, so that load drift spreads evenly over the regimes. Report medians.

Usage:
    python run_timing.py --reps 10
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from src.config import DEVICE, MODEL_TYPES, SEARCH_EPOCHS, set_seeds
from src.data import build_dataset, prepare_splits
from src.evaluation import predict
from src.training import build_model, train_model
from src.utils import code_fingerprint, now, prepare_out_dir, versions, write_json

PARAMS = {'num_layers': 3, 'hidden_size': 256, 'base_rate': 0.2, 'alpha': 1.0}
REGIMES = [('none', 'none'), ('standard', 'none'), ('monte_carlo', 'none'),
           ('fixed', 'tv'), ('dynamic', 'tv'), ('adaptive', 'tv')]


def main():
    ap = argparse.ArgumentParser(description="Controlled timing of dropout regimes")
    ap.add_argument('--config', default='gold_2010_2025')
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--out-root', default='results_timing')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    out = prepare_out_dir(args.out_root, args.overwrite)

    df, X = build_dataset(args.config)
    data = prepare_splits(df, X)
    tr, va = data['train'], data['val']

    def fit(arch, dt, dm, epochs):
        model = build_model(arch, PARAMS, dt, dm, tr['X'].shape[2], DEVICE)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_model(model, tr['X'], tr['y_s'], epochs, DEVICE, monitor=va)
        predict(model, tr['X'], data['scaler_y'], DEVICE)
        predict(model, va['X'], data['scaler_y'], DEVICE)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    for arch in MODEL_TYPES:                      # untimed warm-up
        set_seeds(0)
        fit(arch, 'none', 'none', 1)
    pairs = [(arch, dt, dm) for arch in MODEL_TYPES for dt, dm in REGIMES]
    rows = []
    for rep in range(args.reps):
        order = np.random.default_rng(rep).permutation(len(pairs))
        for pos, k in enumerate(order):
            arch, dt, dm = pairs[k]
            set_seeds(100 + rep)
            sec = fit(arch, dt, dm, SEARCH_EPOCHS)
            rows.append({'model': arch, 'regime': dt, 'divergence': dm, 'rep': rep, 'position': pos, 'seconds': sec})
            print(f"rep {rep} #{pos:2d} {arch:4s} {dt:12s}: {sec:.2f} s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, 'timing.csv'), index=False)
    write_json(os.path.join(out, 'run_info.json'), {'args': vars(args), 'params': PARAMS, 'epochs': SEARCH_EPOCHS,
                                                    'finished': now(), 'versions': versions(),
                                                    'code_md5': code_fingerprint()})
    print(f"✅ {out}/timing.csv")


if __name__ == '__main__':
    main()
