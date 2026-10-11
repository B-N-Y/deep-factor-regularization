#!/usr/bin/env python
"""
run_grid_search.py
------------------
Main experiment: for one test asset, sample period and architecture, one
Optuna study per dropout regime x divergence.

    combos: none, monte_carlo, standard (once each) + {fixed, dynamic, adaptive} x divergences
            + the comparison regimes weight_decay and early_stopping (no dropout)
    search: each trial trains on the training block and is scored by validation RMSE
    refit : the best trial is retrained on train + validation and evaluated once on test
    model selection (here and in every later script) uses validation RMSE only
    benchmark: the linear time-series factor regression on the same day-t inputs

Usage:
    python run_grid_search.py --config c_2010_2025 --model lstm
    python run_grid_search.py --config cl_2010_2025 --model mlp --divergences tv,kl,hellinger,wasserstein
    python run_grid_search.py --config gc_2015_2020 --model gru --n-trials 2 --search-epochs 1 \
        --retrain-epochs 1 --dropout fixed --divergences tv --out-root smoke      # quick check
"""
import argparse
import os

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from src.config import (ALT_REGULARIZERS, DEVICE, DIVERGENCE_METHODS, EARLY_STOPPING_PATIENCE, EXPERIMENTS,
                        MODEL_TYPES, N_TRIALS, PENALIZED_TYPES, RETRAIN_EPOCHS, SEARCH_EPOCHS, SEARCH_SPACE,
                        SELECTION_EXCLUDES, set_seeds)
from src.data import assert_target_not_in_inputs, build_dataset, describe_splits, input_cols, prepare_splits
from src.evaluation import LINEAR, benchmark_fits, metrics, predict, summarize_fit
from src.training import NonFiniteLoss, build_model, free_memory, is_out_of_memory, train_model
from src.utils import code_fingerprint, now, prepare_out_dir, versions, write_json


def make_combos(dropouts: list = None, divergences: list = None) -> list:
    divs = divergences or DIVERGENCE_METHODS
    combos = [('none', 'none'), ('monte_carlo', 'none'), ('standard', 'none')]
    combos += [(dt, dm) for dt in PENALIZED_TYPES for dm in divs]
    combos += [(dt, 'none') for dt in ALT_REGULARIZERS]
    if dropouts:
        combos = [c for c in combos if c[0] in dropouts]
    if not combos:
        raise SystemExit(f"No combination matches dropout={dropouts}, divergences={divergences}")
    return combos


def suggest_params(trial, dropout_type: str, alpha_scale: str = 'linear') -> dict:
    lo, hi = SEARCH_SPACE['num_layers']
    params = {
        'num_layers': trial.suggest_int('num_layers', lo, hi),
        'hidden_size': trial.suggest_categorical('hidden_size', SEARCH_SPACE['hidden_size']),
    }
    if dropout_type != 'none' and dropout_type not in ALT_REGULARIZERS:
        params['base_rate'] = trial.suggest_float('base_rate', *SEARCH_SPACE['base_rate'])
    if dropout_type == 'weight_decay':
        params['weight_decay'] = trial.suggest_float('weight_decay', *SEARCH_SPACE['weight_decay'], log=True)
    if dropout_type in PENALIZED_TYPES:
        if alpha_scale == 'log':
            params['alpha'] = trial.suggest_float('alpha', *SEARCH_SPACE['alpha_log'], log=True)
        else:
            a_lo, a_hi, a_step = SEARCH_SPACE['alpha']
            params['alpha'] = trial.suggest_float('alpha', a_lo, a_hi, step=a_step)
    return params


def alt_kwargs(dt: str, params: dict, val: dict = None) -> dict:
    """Training arguments of the comparison regimes; empty for every dropout regime."""
    if dt == 'weight_decay':
        return {'weight_decay': params['weight_decay']}
    if dt == 'early_stopping' and val is not None:
        return {'early_stop': {'X': val['X'], 'y_s': val['y_s'], 'patience': EARLY_STOPPING_PATIENCE}}
    return {}


def run_search(arch: str, dt: str, dm: str, data: dict, n_trials: int, epochs: int, seed: int = 42,
               alpha_scale: str = 'linear'):
    """Optuna study for one combo. Returns (study, trial rows, best trial artefacts or None)."""
    tr, va = data['train'], data['val']
    n_in = tr['X'].shape[2]
    keep = {}

    def objective(trial):
        set_seeds(seed + trial.number)
        trial.set_user_attr('seed', seed + trial.number)
        params = suggest_params(trial, dt, alpha_scale)
        try:
            model = build_model(arch, params, dt, dm, n_in, DEVICE)
            hist = train_model(model, tr['X'], tr['y_s'], epochs, DEVICE, monitor=va, **alt_kwargs(dt, params, va))
            pred_tr = predict(model, tr['X'], data['scaler_y'], DEVICE)
            pred_va = predict(model, va['X'], data['scaler_y'], DEVICE)
        except Exception as err:  # noqa: BLE001
            if not (isinstance(err, NonFiniteLoss) or is_out_of_memory(err)):
                raise
            # Optuna records a NaN objective as a FAILED trial and continues the study
            reason = 'out of memory' if is_out_of_memory(err) else str(err)
            trial.set_user_attr('failure', reason)
            print(f"   ⚠️  trial {trial.number} failed: {reason}", flush=True)
            model = None
            free_memory()
            return float('nan')
        m_tr, m_va = metrics(tr['y'], pred_tr), metrics(va['y'], pred_va)
        keep[trial.number] = {'val_pred': pred_va, 'history': hist}
        trial.set_user_attr('train', m_tr)
        trial.set_user_attr('val', m_va)
        trial.set_user_attr('eff_rate', hist['rate'][-1])
        if 'best_epoch' in hist:
            trial.set_user_attr('es_epochs', hist['best_epoch'])
        return m_va['RMSE']

    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    rows = []
    for t in study.trials:
        ok = t.state == optuna.trial.TrialState.COMPLETE
        a = t.user_attrs
        rows.append({
            'Model': arch, 'Dropout_Type': dt, 'Divergence': dm, 'Trial': t.number, 'Seed': a['seed'],
            'State': t.state.name, 'Failure': a.get('failure', ''),
            'Num_Layers': t.params['num_layers'], 'Hidden_Size': t.params['hidden_size'],
            'Base_Rate': t.params.get('base_rate', 0.0), 'Alpha': t.params.get('alpha', 0.0),
            'Weight_Decay': t.params.get('weight_decay', 0.0), 'ES_Epochs': a.get('es_epochs', np.nan),
            'Effective_Rate': a['eff_rate'] if ok else np.nan,
            'Train_RMSE': a['train']['RMSE'] if ok else np.nan, 'Train_R2': a['train']['R2'] if ok else np.nan,
            'Val_RMSE': a['val']['RMSE'] if ok else np.nan, 'Val_R2': a['val']['R2'] if ok else np.nan,
            'Gen_Gap': a['val']['RMSE'] - a['train']['RMSE'] if ok else np.nan,
        })
    if not keep:            # every trial failed
        return study, rows, None
    return study, rows, keep[study.best_trial.number]


def refit_and_test(arch: str, params: dict, dt: str, dm: str, data: dict, epochs: int,
                   seed: int = 42, attempts: int = 3):
    """
    Retrain on train + validation with the selected hyperparameters, predict test.
    If the refit breaks down numerically, retry with seed+1, seed+2.
    Returns (test predictions, history, seed used); (None, None, None) if every attempt
    fails or the configuration does not fit in memory.
    """
    tr, va, te = data['train'], data['val'], data['test']
    if dt == 'early_stopping':     # the epoch found on the validation block, scaled like every other regime's refit
        epochs = max(1, int(round(params['es_epochs'] * RETRAIN_EPOCHS / SEARCH_EPOCHS)))
    for s in range(seed, seed + attempts):
        set_seeds(s)
        try:
            model = build_model(arch, params, dt, dm, tr['X'].shape[2], DEVICE)
            hist = train_model(model, np.concatenate([tr['X'], va['X']]),
                               np.concatenate([tr['y_s'], va['y_s']]), epochs, DEVICE, **alt_kwargs(dt, params))
            return predict(model, te['X'], data['scaler_y'], DEVICE), hist, s
        except Exception as err:  # noqa: BLE001
            if is_out_of_memory(err):
                print("   ⚠️  refit out of memory", flush=True)
                free_memory()
                return None, None, None
            if not isinstance(err, NonFiniteLoss):
                raise
            print(f"   ⚠️  refit with seed {s} failed: {err}", flush=True)
    return None, None, None


def refit_selected(arch: str, study, dt: str, dm: str, data: dict, epochs: int, seed: int = 42,
                   max_candidates: int = 3):
    """
    Refit the best trial; if it cannot be refit (numerical breakdown on every seed,
    or out of memory), fall back to the next-best completed trial.
    Returns (trial used, predictions, history, refit seed) or (None, ...) if all fail.
    """
    done = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                  key=lambda t: t.value)
    for t in done[:max_candidates]:
        params = dict(t.params)
        if 'es_epochs' in t.user_attrs:
            params['es_epochs'] = t.user_attrs['es_epochs']
        pred, hist, s = refit_and_test(arch, params, dt, dm, data, epochs, seed)
        if pred is not None:
            return t, pred, hist, s
        print(f"   ⚠️  trial {t.number} could not be refit; trying the next-best trial", flush=True)
    return None, None, None, None


def main():
    parser = argparse.ArgumentParser(description="Dropout x divergence grid search for deep factor models")
    parser.add_argument('--config', required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument('--model', required=True, choices=MODEL_TYPES)
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--search-epochs', type=int, default=SEARCH_EPOCHS)
    parser.add_argument('--retrain-epochs', type=int, default=RETRAIN_EPOCHS)
    parser.add_argument('--dropout', default=None, help="Comma-separated subset of dropout types")
    parser.add_argument('--divergences', default=None, help="Comma-separated subset of divergences")
    parser.add_argument('--alpha-scale', default='linear', choices=['linear', 'log'],
                        help="Search the divergence weight on the default grid or log-uniformly on [1e-3, 1e3]")
    parser.add_argument('--part', default=None,
                        help="Name of this part when one experiment is split over several jobs; "
                             "parts are combined with merge_results.py")
    parser.add_argument('--out-root', default='results')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    divs = args.divergences.split(',') if args.divergences else None
    if divs and not set(divs) <= set(DIVERGENCE_METHODS):
        raise SystemExit(f"Unknown divergence(s): {sorted(set(divs) - set(DIVERGENCE_METHODS))}")
    dropouts = args.dropout.split(',') if args.dropout else None
    exp = EXPERIMENTS[args.config]
    name = args.config + (f"__part_{args.part}" if args.part else '')
    out_dir = prepare_out_dir(os.path.join(args.out_root, args.model, name), args.overwrite)
    started = now()

    print("=" * 70)
    print(f"Grid search | {args.config} | {exp['name']} | model={args.model} | device={DEVICE}")
    n_checked = assert_target_not_in_inputs(args.config)
    print(f"🔒 Own-return check passed on {n_checked} perturbed days (inputs: {input_cols(exp['asset'])})")

    df, X = build_dataset(args.config)
    data = prepare_splits(df, X)
    print(describe_splits(data))
    combos = make_combos(dropouts, divs)
    print(f"📊 {len(combos)} combos x {args.n_trials} trials | search {args.search_epochs} ep | refit {args.retrain_epochs} ep")
    print("=" * 70)

    te = data['test']
    fit_idx = np.concatenate([data['train']['idx'], data['val']['idx']])
    X_t = X[:, -1, :]
    y_all = df['target'].values
    bench = benchmark_fits(X_t, y_all, fit_idx, te['idx'])
    linear = bench[LINEAR]
    pd.DataFrame([{'Benchmark': b, **summarize_fit(te['y'], p, linear)} for b, p in bench.items()]) \
        .to_csv(os.path.join(out_dir, 'benchmarks.csv'), index=False)

    all_rows, best_rows = [], []
    preds = {'meta': {'asset': exp['asset'], 'linear': LINEAR,
                      'dates_val': data['val']['dates'].tolist(), 'y_val': data['val']['y'].tolist(),
                      'dates_test': te['dates'].tolist(), 'y_test': te['y'].tolist(),
                      # raw return and rf on the test days, to rebuild price levels:
                      # P_hat_t = P_{t-1} * (1 + (fitted excess return_t + rf_t) / 100)
                      'ret_test': df['ret'].values[te['idx']].tolist(),
                      'rf_test': df['rf'].values[te['idx']].tolist()},
             'benchmarks': {k: v.tolist() for k, v in bench.items()}, 'models': {}}
    histories = {}

    for i, (dt, dm) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] {args.model}: {dt} + {dm}", flush=True)
        study, rows, best_art = run_search(args.model, dt, dm, data, args.n_trials, args.search_epochs,
                                           alpha_scale=args.alpha_scale)
        all_rows.extend(rows)
        if best_art is None:
            print(f"   ❌ {dt}+{dm}: every trial failed; combination reported as missing", flush=True)
            pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, 'all_trials.csv'), index=False)
            continue
        n_failed = sum(t.state != optuna.trial.TrialState.COMPLETE for t in study.trials)
        best, pred_te, hist_refit, refit_seed = refit_selected(args.model, study, dt, dm, data, args.retrain_epochs)
        if best is None:
            print(f"   ❌ {dt}+{dm}: no trial could be refit; combination reported as missing", flush=True)
            continue
        if best.number != study.best_trial.number:
            best_art = None          # validation predictions stored only for the best trial
        params = dict(best.params)
        summ = summarize_fit(te['y'], pred_te, linear)
        key = f"{dt}__{dm}"
        preds['models'][key] = {'val': best_art['val_pred'].tolist() if best_art else None, 'test': pred_te.tolist()}
        histories[key] = {'search_best': best_art['history'] if best_art else None, 'refit': hist_refit}

        best_rows.append({
            'Model': args.model, 'Dropout_Type': dt, 'Divergence': dm, 'Best_Trial': best.number,
            'Refit_Fallback': best.number != study.best_trial.number,
            'N_Failed_Trials': n_failed, 'Refit_Seed': refit_seed,
            'Num_Layers': params['num_layers'], 'Hidden_Size': params['hidden_size'],
            'Base_Rate': params.get('base_rate', 0.0), 'Alpha': params.get('alpha', 0.0),
            'Weight_Decay': params.get('weight_decay', 0.0), 'ES_Epochs': best.user_attrs.get('es_epochs', np.nan),
            'Effective_Rate_Refit': hist_refit['rate'][-1],
            'Train_RMSE': best.user_attrs['train']['RMSE'], 'Train_R2': best.user_attrs['train']['R2'],
            'Val_RMSE': best.user_attrs['val']['RMSE'], 'Val_R2': best.user_attrs['val']['R2'],
            'Gen_Gap': best.user_attrs['val']['RMSE'] - best.user_attrs['train']['RMSE'],
            **{f'Test_{k}': v for k, v in summ.items()},
        })
        print(f"   val RMSE {best.user_attrs['val']['RMSE']:.4f} | test R2 {summ['R2']:+.4f} (OLS factors "
              f"{1 - np.sum((te['y'] - linear) ** 2) / np.sum((te['y'] - te['y'].mean()) ** 2):+.4f}) "
              f"| DM vs OLS {summ['DM_vs_ols']:+.2f}", flush=True)

        # write after every combo so a timeout never loses finished work
        pd.DataFrame(best_rows).to_csv(os.path.join(out_dir, 'best_per_combo.csv'), index=False)
        pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, 'all_trials.csv'), index=False)
        write_json(os.path.join(out_dir, 'predictions.json'), preds)
        write_json(os.path.join(out_dir, 'training_history.json'), histories)

    best_df = pd.DataFrame(best_rows)
    sel_all = best_df.loc[best_df['Val_RMSE'].idxmin()]
    drop_df = best_df[~best_df['Dropout_Type'].isin(SELECTION_EXCLUDES)]
    sel_drop = drop_df.loc[drop_df['Val_RMSE'].idxmin()] if len(drop_df) else None
    write_json(os.path.join(out_dir, 'selection.json'), {
        'rule': 'lowest validation RMSE; test set never used for selection',
        'best_overall': {'Dropout_Type': sel_all['Dropout_Type'], 'Divergence': sel_all['Divergence'],
                         'Val_RMSE': sel_all['Val_RMSE'], 'Test_RMSE': sel_all['Test_RMSE']},
        'best_dropout': None if sel_drop is None else {
            'Dropout_Type': sel_drop['Dropout_Type'], 'Divergence': sel_drop['Divergence'],
            'Val_RMSE': sel_drop['Val_RMSE'], 'Test_RMSE': sel_drop['Test_RMSE']},
    })
    write_json(os.path.join(out_dir, 'run_info.json'), {
        'config': args.config, 'model': args.model, 'experiment': exp, 'args': vars(args),
        'inputs': input_cols(exp['asset']), 'window': int(X.shape[1]),
        'started': started, 'finished': now(), 'device': DEVICE,
        'own_return_days_checked': n_checked,
        'splits': {p: {'start': data[p]['dates'][0], 'end': data[p]['dates'][-1], 'n': len(data[p]['dates'])}
                   for p in ('train', 'val', 'test')},
        'versions': versions(), 'code_md5': code_fingerprint(),
    })

    print("\n" + "=" * 70)
    print(f"★ Selected by validation (all models): {sel_all['Dropout_Type']} + {sel_all['Divergence']}")
    if sel_drop is not None:
        print(f"★ Selected by validation (dropout):    {sel_drop['Dropout_Type']} + {sel_drop['Divergence']}")
    print(f"✅ Results in {out_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
