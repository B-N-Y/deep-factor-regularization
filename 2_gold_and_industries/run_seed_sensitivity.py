#!/usr/bin/env python
"""
run_seed_sensitivity.py
-----------------------
Repeats the full Optuna search + refit with a different base seed for
    (a) the dropout configuration selected by validation RMSE in the main run, and
    (b) the same architecture without dropout.
The selection is read from <main-root>/<model>/<config>/selection.json (validation only).

Usage:
    python run_seed_sensitivity.py --config gold_2010_2025 --model lstm --seed 123
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from run_grid_search import refit_and_test, run_search
from src.config import EXPERIMENTS, MODEL_TYPES, N_TRIALS, RETRAIN_EPOCHS, SEARCH_EPOCHS
from src.data import assert_target_not_in_inputs, build_dataset, prepare_splits
from src.evaluation import LINEAR, benchmark_fits, summarize_fit
from src.utils import code_fingerprint, now, prepare_out_dir, versions, write_json


def main():
    parser = argparse.ArgumentParser(description="Seed sensitivity")
    parser.add_argument('--config', required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument('--model', required=True, choices=MODEL_TYPES)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--search-epochs', type=int, default=SEARCH_EPOCHS)
    parser.add_argument('--retrain-epochs', type=int, default=RETRAIN_EPOCHS)
    parser.add_argument('--main-root', default='results')
    parser.add_argument('--out-root', default='results_seed')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    with open(os.path.join(args.main_root, args.model, args.config, 'selection.json')) as f:
        sel = json.load(f)['best_dropout']
    combos = [(sel['Dropout_Type'], sel['Divergence']), ('none', 'none')]
    out_dir = prepare_out_dir(os.path.join(args.out_root, args.model, args.config, f'seed_{args.seed}'), args.overwrite)
    started = now()

    print(f"Seed sensitivity | {args.config} | {args.model} | seed={args.seed} | combos={combos}")
    assert_target_not_in_inputs(args.config)
    df, X = build_dataset(args.config)
    data = prepare_splits(df, X)
    te = data['test']
    fit_idx = np.concatenate([data['train']['idx'], data['val']['idx']])
    linear = benchmark_fits(X[:, -1, :], df['target'].values, fit_idx, te['idx'])[LINEAR]

    rows, preds = [], {}
    for dt, dm in combos:
        study, _, best_art = run_search(args.model, dt, dm, data, args.n_trials, args.search_epochs, seed=args.seed)
        if best_art is None:
            print(f"   ❌ {dt}+{dm}: every trial failed")
            continue
        best = study.best_trial
        params = dict(best.params)
        n_failed = sum(t.state.name != 'COMPLETE' for t in study.trials)
        pred_te, _, refit_seed = refit_and_test(args.model, params, dt, dm, data, args.retrain_epochs, seed=args.seed)
        if pred_te is None:
            print(f"   ❌ {dt}+{dm}: every refit attempt failed")
            continue
        summ = summarize_fit(te['y'], pred_te, linear)
        preds[f'{dt}__{dm}'] = pred_te.tolist()
        rows.append({'Seed': args.seed, 'Model': args.model, 'Dropout_Type': dt, 'Divergence': dm,
                     'N_Failed_Trials': n_failed, 'Refit_Seed': refit_seed,
                     'Num_Layers': params['num_layers'], 'Hidden_Size': params['hidden_size'],
                     'Base_Rate': params.get('base_rate', 0.0), 'Alpha': params.get('alpha', 0.0),
                     'Train_R2': best.user_attrs['train']['R2'], 'Val_R2': best.user_attrs['val']['R2'],
                     'Val_RMSE': best.value, **{f'Test_{k}': v for k, v in summ.items()}})
        print(f"   {dt}+{dm}: val RMSE {best.value:.4f} | test R2 {summ['R2']:+.4f} | vs OLS {summ['R2_vs_ols']:+.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'seed_result.csv'), index=False)
    write_json(os.path.join(out_dir, 'predictions.json'), {'dates_test': te['dates'].tolist(),
                                                           'y_test': te['y'].tolist(), 'models': preds})
    write_json(os.path.join(out_dir, 'run_info.json'), {
        'config': args.config, 'model': args.model, 'args': vars(args), 'selection_used': sel,
        'started': started, 'finished': now(), 'versions': versions(), 'code_md5': code_fingerprint()})
    print(f"✅ Results in {out_dir}/")


if __name__ == '__main__':
    main()
