#!/usr/bin/env python
"""
merge_results.py
----------------
Combines the parts of one experiment that was split over several jobs
(results/<model>/<config>__part_<name>/) into results/<model>/<config>/ and
writes selection.json there. Refuses to merge if a part is unfinished, if the
parts disagree on data or benchmarks, or if any combination is missing.

Usage:
    python merge_results.py --config gold_2010_2025 --model lstm
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from run_grid_search import make_combos
from src.config import EXPERIMENTS, MODEL_TYPES, SELECTION_EXCLUDES
from src.utils import code_fingerprint, code_mismatch, now, prepare_out_dir, same_run_data, write_json


def main():
    ap = argparse.ArgumentParser(description="Merge split experiment parts")
    ap.add_argument('--config', required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument('--model', required=True, choices=MODEL_TYPES)
    ap.add_argument('--root', default='results')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    parts = sorted(glob.glob(os.path.join(args.root, args.model, f'{args.config}__part_*')))
    if not parts:
        raise SystemExit("❌ no parts found")
    for p in parts:
        if not os.path.isfile(os.path.join(p, 'run_info.json')):
            raise SystemExit(f"❌ part not finished: {p}")

    load = lambda p, f: json.load(open(os.path.join(p, f)))  # noqa: E731
    infos = [load(p, 'run_info.json') for p in parts]
    preds = [load(p, 'predictions.json') for p in parts]
    hists = [load(p, 'training_history.json') for p in parts]

    bad = code_mismatch([i['code_md5'] for i in infos])
    if bad:
        raise SystemExit(f"❌ parts were produced by different versions of {bad}")
    for p in preds[1:]:
        if not same_run_data(p, preds[0]):
            raise SystemExit("❌ parts disagree on data split or benchmarks")
    bench = [pd.read_csv(os.path.join(p, 'benchmarks.csv')) for p in parts]
    if any(not np.allclose(b.select_dtypes('number').values, bench[0].select_dtypes('number').values, equal_nan=True)
           for b in bench[1:]):
        raise SystemExit("❌ parts disagree on benchmarks.csv")

    best = pd.concat([pd.read_csv(os.path.join(p, 'best_per_combo.csv')) for p in parts], ignore_index=True)
    trials = pd.concat([pd.read_csv(os.path.join(p, 'all_trials.csv')) for p in parts], ignore_index=True)
    # every combination that the parts were asked to run (from their own command-line arguments)
    expected = set()
    for i in infos:
        a = i['args']
        expected |= set(make_combos(a['dropout'].split(',') if a.get('dropout') else None,
                                    a['divergences'].split(',') if a.get('divergences') else None))
    have = set(zip(best['Dropout_Type'], best['Divergence']))
    if have != expected or best.duplicated(['Dropout_Type', 'Divergence']).any():
        raise SystemExit(f"❌ combinations missing {sorted(expected - have)} or duplicated")

    out = prepare_out_dir(os.path.join(args.root, args.model, args.config), args.overwrite)
    best.to_csv(os.path.join(out, 'best_per_combo.csv'), index=False)
    trials.to_csv(os.path.join(out, 'all_trials.csv'), index=False)
    bench[0].to_csv(os.path.join(out, 'benchmarks.csv'), index=False)
    merged_pred = {'meta': preds[0]['meta'], 'benchmarks': preds[0]['benchmarks'],
                   'models': {k: v for p in preds for k, v in p['models'].items()}}
    write_json(os.path.join(out, 'predictions.json'), merged_pred)
    write_json(os.path.join(out, 'training_history.json'), {k: v for h in hists for k, v in h.items()})

    sel_all = best.loc[best['Val_RMSE'].idxmin()]
    drop = best[~best['Dropout_Type'].isin(SELECTION_EXCLUDES)]
    sel_drop = drop.loc[drop['Val_RMSE'].idxmin()]
    write_json(os.path.join(out, 'selection.json'), {
        'rule': 'lowest validation RMSE; test set never used for selection',
        'best_overall': {'Dropout_Type': sel_all['Dropout_Type'], 'Divergence': sel_all['Divergence'],
                         'Val_RMSE': sel_all['Val_RMSE'], 'Test_RMSE': sel_all['Test_RMSE']},
        'best_dropout': {'Dropout_Type': sel_drop['Dropout_Type'], 'Divergence': sel_drop['Divergence'],
                         'Val_RMSE': sel_drop['Val_RMSE'], 'Test_RMSE': sel_drop['Test_RMSE']},
    })
    info = dict(infos[0])
    info.update({'merged_from': [os.path.basename(p) for p in parts], 'merged_at': now(),
                 'started': min(i['started'] for i in infos), 'finished': max(i['finished'] for i in infos),
                 'parts': infos, 'merge_code_md5': code_fingerprint()})
    write_json(os.path.join(out, 'run_info.json'), info)
    print(f"✅ merged {len(parts)} parts ({len(best)} combos, {len(trials)} trials) into {out}")


if __name__ == '__main__':
    main()
