#!/usr/bin/env python
"""
run_expanding_cv.py
-------------------
Expanding-window evaluation (robustness). With TimeSeriesSplit(n_folds), fold k
trains on blocks 1..k and is evaluated on block k+1. In every fold the
architecture is refit with the hyperparameters chosen in the main run for
    (a) the dropout configuration selected by validation RMSE and (b) no dropout,
and compared with the linear factor regression re-estimated on the fold's
training block. Scalers are re-estimated on each fold's training block.

Note: hyperparameters come from the main run, whose validation block lies inside
the later folds; the folds are therefore a robustness check, not a fresh test.

Usage:
    python run_expanding_cv.py --config c_2010_2025 --model lstm --n-folds 5
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import DEVICE, EXPERIMENTS, MODEL_TYPES, RETRAIN_EPOCHS, set_seeds
from src.data import assert_target_not_in_inputs, build_dataset, scale_arrays
from src.evaluation import LINEAR, benchmark_fits, dm_test, predict, summarize_fit
from src.training import NonFiniteLoss, build_model, train_model
from src.utils import code_fingerprint, now, prepare_out_dir, versions, write_json


def main():
    parser = argparse.ArgumentParser(description="Expanding-window evaluation")
    parser.add_argument('--config', required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument('--model', required=True, choices=MODEL_TYPES)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=RETRAIN_EPOCHS)
    parser.add_argument('--main-root', default='results')
    parser.add_argument('--out-root', default='results_expanding_cv')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    main_dir = os.path.join(args.main_root, args.model, args.config)
    with open(os.path.join(main_dir, 'selection.json')) as f:
        sel = json.load(f)['best_dropout']
    best = pd.read_csv(os.path.join(main_dir, 'best_per_combo.csv'))
    combos = []
    for dt, dm in [(sel['Dropout_Type'], sel['Divergence']), ('none', 'none')]:
        r = best[(best['Dropout_Type'] == dt) & (best['Divergence'] == dm)].iloc[0]
        combos.append((dt, dm, {'num_layers': int(r['Num_Layers']), 'hidden_size': int(r['Hidden_Size']),
                                'base_rate': float(r['Base_Rate']), 'alpha': float(r['Alpha'])}))
    out_dir = prepare_out_dir(os.path.join(args.out_root, args.model, args.config), args.overwrite)
    started = now()

    print(f"Expanding-window CV | {args.config} | {args.model} | folds={args.n_folds} | combos={[c[:2] for c in combos]}")
    assert_target_not_in_inputs(args.config)
    df, X = build_dataset(args.config)
    y_all = df['target'].values

    rows, fold_preds = [], []
    for k, (tr_idx, te_idx) in enumerate(TimeSeriesSplit(n_splits=args.n_folds).split(np.arange(len(df))), 1):
        data = scale_arrays(df, X, tr_idx, {'train': tr_idx, 'test': te_idx})
        tr, te = data['train'], data['test']
        bench = benchmark_fits(X[:, -1, :], y_all, tr_idx, te_idx)
        linear = bench[LINEAR]
        preds = {}
        for dt, dm, params in combos:
            for s in (42 + k, 1042 + k, 2042 + k):     # retry only if training breaks down numerically
                set_seeds(s)
                model = build_model(args.model, params, dt, dm, tr['X'].shape[2], DEVICE)
                try:
                    train_model(model, tr['X'], tr['y_s'], args.epochs, DEVICE)
                except NonFiniteLoss as err:
                    print(f"   ⚠️  fold {k} {dt}+{dm} seed {s} failed: {err}", flush=True)
                    continue
                preds[f'{dt}__{dm}'] = predict(model, te['X'], data['scaler_y'], DEVICE)
                break
            else:
                raise SystemExit(f"❌ fold {k} {dt}+{dm}: every attempt failed")

        for key, yhat in {**preds, **{f'benchmark:{b}': v for b, v in bench.items()}}.items():
            rows.append({'Fold': k, 'Train_Start': tr['dates'][0], 'Train_End': tr['dates'][-1],
                         'Test_Start': te['dates'][0], 'Test_End': te['dates'][-1], 'N_Test': len(te_idx),
                         'Model': key, **summarize_fit(te['y'], yhat, linear)})
        k_drop, k_none = f'{combos[0][0]}__{combos[0][1]}', 'none__none'
        dm_res = dm_test(te['y'] - preds[k_none], te['y'] - preds[k_drop])
        rows.append({'Fold': k, 'Model': f'DM {k_drop} vs none', 'DM_vs_ols': dm_res['DM_stat'],
                     'DM_vs_ols_p': dm_res['DM_pvalue']})
        fold_preds.append({'fold': k, 'dates': te['dates'].tolist(), 'y': te['y'].tolist(),
                           'models': {m: v.tolist() for m, v in preds.items()},
                           'benchmarks': {b: v.tolist() for b, v in bench.items()}})
        print(f"   fold {k}: {te['dates'][0]} -> {te['dates'][-1]} done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, 'fold_results.csv'), index=False)
    summary = (res[~res['Model'].str.startswith('DM ')]
               .groupby('Model')[['RMSE', 'R2', 'R2_vs_ols']].agg(['mean', 'std']))
    summary.to_csv(os.path.join(out_dir, 'cv_summary.csv'))
    write_json(os.path.join(out_dir, 'predictions.json'), fold_preds)
    write_json(os.path.join(out_dir, 'run_info.json'), {
        'config': args.config, 'model': args.model, 'args': vars(args), 'selection_used': sel,
        'combos': [{'dropout': c[0], 'divergence': c[1], 'params': c[2]} for c in combos],
        'started': started, 'finished': now(), 'versions': versions(), 'code_md5': code_fingerprint()})
    print(summary.round(4).to_string())
    print(f"✅ Results in {out_dir}/")


if __name__ == '__main__':
    main()
