#!/usr/bin/env python
"""
validate_returns.py
-------------------
Checks the daily futures returns of futures_data.csv against LSEG's own settlement
change (SETL_PCHNG, in %) of the continuous RICs <root>c1..c3 (data/raw_lseg/).

A contract-day is checked when the contract held on day t (futures_held.csv) sits in
the same continuous slot on t and on the slot's previous LSEG trading day, so that
SETL_PCHNG refers to the same contract as our return, when that previous LSEG trading
day is also the previous NYSE day, so that both changes cover the same interval, and
when both values are available. (Our returns follow the NYSE calendar: when the NYSE is
closed and the exchange is open, the next NYSE return spans two exchange sessions.)
SETL_PCHNG is reported by LSEG from 2018 for some roots and from 2021-2023 for the
others. A deviation is a difference larger than TOL percentage points. The two deviations
(ICE Brent and gasoil, 4 January 2022) arise because LSEG measures that day's change from
the settlement of 31 December 2021 and skips the settlement of 3 January, a UK bank holiday
on which the contracts traded and settled.

    python data/validate_returns.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LSEG = os.path.join(HERE, 'raw_lseg')
TOL = 1e-4          # percentage points; futures_data.csv stores six decimals


def parse_expiry(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors='coerce')
    num = num[(num >= 19000101) & (num <= 21001231)]
    return pd.to_datetime(num.dropna().astype('int64').astype(str), format='%Y%m%d').reindex(s.index)


def validate():
    """Number of checked contract-days, first and last day, roots covered, and the deviations."""
    data = pd.read_csv(os.path.join(HERE, 'futures_data.csv'), index_col='Date', parse_dates=True)
    held = pd.read_csv(os.path.join(HERE, 'futures_held.csv'), index_col='Date', parse_dates=True)
    held = held.apply(pd.to_datetime)
    roots = [c[4:] for c in data.columns if c.startswith('ret_')]
    rows, devs = [], []
    for root in roots:
        checked = set()
        for n in (1, 2, 3):
            f = os.path.join(LSEG, f'{root}c{n}.csv')
            if not os.path.isfile(f):
                continue
            d = pd.read_csv(f, index_col='Date', parse_dates=True)
            d = d[pd.to_numeric(d['SETTLE'], errors='coerce').notna()]
            label = parse_expiry(d['EXPIR_DATE'])
            pch = pd.to_numeric(d['SETL_PCHNG'], errors='coerce')
            same = label.notna() & (label == label.shift())        # same contract as on the previous LSEG day
            prev_lseg = pd.Series(d.index, index=d.index).shift()
            for t in d.index[same & pch.notna()]:
                if t not in data.index or t in checked or pd.isna(data.at[t, f'ret_{root}']):
                    continue
                k = data.index.get_loc(t)
                if k == 0 or data.index[k - 1] != prev_lseg[t]:      # the two changes would cover different intervals
                    continue
                if held.at[t, root] != label[t]:
                    continue
                checked.add(t)
                diff = data.at[t, f'ret_{root}'] - pch[t]
                rows.append((root, t))
                if abs(diff) > TOL:
                    devs.append((root, t.date(), round(data.at[t, f'ret_{root}'], 4), round(pch[t], 4)))
    return dict(n=len(rows), roots=len({r for r, _ in rows}), first=min(t for _, t in rows).date(),
                last=max(t for _, t in rows).date(), deviations=devs)


def main():
    v = validate()
    rows, devs = v['n'], v['deviations']
    print(f"checked contract-days: {rows}  (roots: {v['roots']}, {v['first']} to {v['last']})")
    print(f"deviations above {TOL} pp: {len(devs)}")
    for dv in devs:
        print('   ', dv)


if __name__ == '__main__':
    main()
