#!/usr/bin/env python
"""
build_futures.py
----------------
Builds the daily commodity-futures dataset from the LSEG downloads in
data/raw_lseg/ (fetch_futures.py) and the public macro series in data/raw/.

futures_data.csv      (calendar = NYSE trading days, 2009-2025; models use 2010-2025)
    ret_<root>          daily excess return of the held contract of each commodity, %.
                        A futures position needs no capital, so its price change is an
                        excess return.
    mkt_rf, rf          Fama-French daily market excess return and T-bill rate, %
    d_tips, d_be        changes in 10-year TIPS yield and breakeven inflation, pp (FRED)
    dlog_dxy, dlog_vix  100 x log changes of the dollar index and the VIX (Yahoo)
    avg_<root>          equal-weighted return of all other commodities, %
    carry_<root>        high-minus-low carry tercile portfolio of the other commodities, %
    mom_<root>          high-minus-low 12-month momentum tercile portfolio of the other commodities, %
    The three commodity factors follow Bakshi, Gao and Rossi (2019, Management
    Science). They are built leave-one-out: the factors of root i never contain
    root i, so the test asset's own return of day t never enters its inputs.

futures_held.csv      expiry of the contract held on every day (audit trail).

Contract roll. In calendar month M the position is the nearest contract that
does not expire before the end of M; the return of day t is measured on the
contract held in the month of t, between the settlement prices of t-1 and t of
that same contract, so roll dates contain no price jump. The position never
enters the delivery or expiry month (this also avoids the negative WTI price of
20 April 2020: in April 2020 the June contract is held).

Carry of month M+1 (measured on the last trading day of M) is
    100 x (ln F_held - ln F_next) x 365 / (days between the two expiries),
where held is the contract held in M+1 and next the following contract;
positive carry = backwardation. Carry needs the price of the contract after the
held one at the month end. Heating oil, RBOB gasoline and Brent contracts expire at
the end of a month, so on the last trading day the expiring contract still occupies
c1 and the contract after the held one is c4; for these roots c1..c4 are used
(fetch_futures.SLOTS). Momentum is the cumulative log return of the
held-contract series over the 12 months up to the end of M. Portfolios are
re-sorted at every month end and held, equal-weighted, during the next month.

Usage:  python data/build_futures.py
"""
import glob
import io
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'raw')
LSEG = os.path.join(HERE, 'raw_lseg')
CONTRACTS = os.path.join(LSEG, 'contracts')

SECTORS = {
    'energy':    ['CL', 'LCO', 'NG', 'HO', 'RB', 'LGO'],
    'metals':    ['GC', 'SI', 'HG', 'PL', 'PA'],
    'grains':    ['C', 'W', 'S', 'BO', 'SM', 'KW', 'O'],
    'softs':     ['KC', 'SB', 'CC', 'CT', 'OJ'],
    'livestock': ['LC', 'LH', 'FC'],
}
ROOTS = [r for roots in SECTORS.values() for r in roots]
CONTRACT_ROOTS = ['PL', 'PA', 'OJ', 'O', 'FC']        # built from individual contracts
WARMUP, START, END = '2008-01-01', '2009-01-01', '2025-12-31'   # START: first day written (windows need pre-2010 days)
FFILL_LIMIT = 5                                        # trading days a settlement may be carried over a holiday
FALLBACK_DAYS = []                                     # days on which root_series used the fallback contract
MAX_STALE_DAYS = 5                                     # longest lag of LSEG expiry labels behind a chain roll


# ---------------------------------------------------------------------------- inputs
def parse_expiry(s: pd.Series) -> pd.Series:
    """EXPIR_DATE comes as 20240115, 20240115.0 or '20240115' (0 or blank = unknown)."""
    num = pd.to_numeric(s, errors='coerce')
    num = num[(num >= 19000101) & (num <= 21001231)]
    return pd.to_datetime(num.dropna().astype('int64').astype(str), format='%Y%m%d').reindex(s.index)


def slots(root: str) -> list:
    """Continuous contracts downloaded for a root: c1..c3, and c4 where fetch_futures.SLOTS asks for it."""
    return [n for n in (1, 2, 3, 4) if n <= 3 or os.path.isfile(os.path.join(LSEG, f'{root}c{n}.csv'))]


def read_continuous(root: str):
    """Settlement and expiry of c1..c3 (c1..c4 for the roots with a fourth slot), index = date."""
    out = {}
    for n in slots(root):
        d = pd.read_csv(os.path.join(LSEG, f'{root}c{n}.csv'), index_col='Date', parse_dates=True)
        out[n] = pd.DataFrame({'settle': pd.to_numeric(d['SETTLE'], errors='coerce'),
                               'oi': pd.to_numeric(d['OPINT_1'], errors='coerce'),
                               'expiry': parse_expiry(d['EXPIR_DATE'])})
    return out


def stale_label_days(c) -> pd.Series:
    """Days on which the continuous chain has already rolled but LSEG's expiry labels have not.

    A roll starts on day t in slot n (1 or 2) when the slot keeps its expiry label
    while its open interest jumps by more than a factor of two to the level of
    yesterday's next slot (within 25%), and its price breaks away from its own
    previous settlement (more than 2% and three times the gap to yesterday's next
    slot). The chain then counts as shifted by one contract for as long as the
    front label stays unchanged, which must end with the label rolling within
    MAX_STALE_DAYS trading days (otherwise the event is not a lagging label and
    is ignored). Lean hogs, 12-13 Feb 2015, is such a case: the chain moved to
    April/May/June two days before the labels did.
    """
    idx = c[1].index
    s = {n: c[n]['settle'].reindex(idx) for n in (1, 2, 3)}
    oi = {n: c[n]['oi'].reindex(idx) for n in (1, 2, 3)}
    start = pd.Series(False, index=idx)
    with np.errstate(divide='ignore', invalid='ignore'):
        for n in (1, 2):
            label = c[n]['expiry'].reindex(idx)
            same = label.notna() & (label == label.shift())
            jump = np.log(oi[n] / oi[n].shift()).abs() > np.log(2)
            match = np.log(oi[n] / oi[n + 1].shift()).abs() < np.log(1.25)
            own = (np.log(s[n]) - np.log(s[n].shift())).abs()
            nxt = (np.log(s[n]) - np.log(s[n + 1].shift())).abs()
            start |= same & jump & match & (own > 0.02) & (own > 3 * nxt)
    e1 = c[1]['expiry'].reindex(idx)
    shifted = pd.Series(False, index=idx)
    for t in idx[start]:
        k0, front = idx.get_loc(t), e1[t]
        k = k0
        while k < len(idx) and e1.iloc[k] == front:
            k += 1
        if k < len(idx) and k - k0 <= MAX_STALE_DAYS:
            shifted.iloc[k0:k] = True
    return shifted


def fill_front_expiry(e: pd.Series) -> pd.Series:
    """Fill gaps in the front-contract expiry with the last known expiry while it has not passed
    (otherwise the day stays unknown and its front-contract price is not used)."""
    ff = e.ffill()
    return ff.where(ff >= e.index.to_series())


def panel_from_continuous(root: str):
    """Contract-level settlement panel (date x expiry) from c1..c3, plus a consistency report.

    Observed expiries are used where LSEG reports them; missing ones are inferred
    from the front contract's position in the chain of expiries. A vendor
    artifact is dropped together with its prices: an expiry whose calendar
    month is not part of the root's regular expiry schedule (months seen in at
    least three years). The only case is sugar: LSEG lists a contract expiring
    2009-12-22, while ICE Sugar No. 11 expiries fall in Feb, Apr, Jun and Sep.
    """
    c = read_continuous(root)
    ns = sorted(c)
    obs = {n: c[n]['expiry'] for n in ns}
    seen = pd.concat([obs[n] for n in ns]).dropna().drop_duplicates()
    years_per_month = seen.groupby(seen.dt.month).apply(lambda e: e.dt.year.nunique())
    regular = set(years_per_month[years_per_month >= 3].index)
    artifacts = set(seen[~seen.dt.month.isin(regular)])
    seen = pd.DataFrame({'expiry': seen})
    valid = {n: ~obs[n].isin(artifacts) for n in ns}

    chain = sorted(set(seen['expiry']) - artifacts)
    pos = {e: k for k, e in enumerate(chain)}
    # front expiry: observed, else n-1 steps before the observed n-th contract, else filled
    e1 = obs[1].where(valid[1])
    for n in ns[1:]:
        k = obs[n].where(valid[n]).map(pos) - (n - 1)
        e1 = e1.fillna(k.map(lambda v: chain[int(v)] if pd.notna(v) and v >= 0 else pd.NaT).reindex(e1.index))
    e1 = fill_front_expiry(e1)
    shifted = stale_label_days(c)
    if shifted.any():
        e1.loc[shifted] = e1[shifted].map(lambda e: chain[pos[e] + 1] if e in pos and pos[e] + 1 < len(chain) else pd.NaT)
        print(f"   {root}: chain rolled before its labels on {[t.date() for t in shifted[shifted].index]}")
    rows, checked, mismatched = [], 0, 0
    for n in ns:
        k = e1.reindex(obs[n].index).map(pos) + (n - 1)
        inferred = k.map(lambda v: chain[int(v)] if pd.notna(v) and v < len(chain) else pd.NaT)
        stale = shifted.reindex(obs[n].index, fill_value=False)
        both = obs[n].notna() & inferred.notna() & valid[n] & ~stale
        checked += int(both.sum())
        mismatched += int((obs[n][both] != inferred[both]).sum())
        expiry = obs[n].where(obs[n].notna() & ~stale, inferred)
        keep = valid[n] & expiry.notna()
        rows.append(pd.DataFrame({'date': expiry.index[keep], 'expiry': expiry[keep].values,
                                  'settle': c[n]['settle'][keep].values}))
    long = pd.concat(rows).dropna(subset=['settle']).drop_duplicates(['date', 'expiry'])
    if artifacts:
        print(f"   {root}: dropped vendor artifact expiries {[e.date() for e in sorted(artifacts)]}")
    return long.pivot(index='date', columns='expiry', values='settle').sort_index(), (checked, mismatched)


def panel_from_contracts(root: str):
    """Contract-level settlement panel from the individual contract files."""
    cols = {}
    for f in sorted(glob.glob(os.path.join(CONTRACTS, f'{root}_*.csv'))):
        d = pd.read_csv(f, index_col='Date', parse_dates=True)
        s = pd.to_numeric(d['SETTLE'], errors='coerce').dropna()
        if s.empty:
            continue
        e = parse_expiry(d['EXPIR_DATE']).dropna()
        expiry = e.iloc[-1] if len(e) else s.index.max()     # last settlement day otherwise
        cols[expiry] = s
    return pd.DataFrame(cols).sort_index().sort_index(axis=1)


def on_calendar(panel: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Each contract's settlement on the NYSE calendar, carried over exchange holidays within its life."""
    out = {}
    for e in panel.columns:
        s = panel[e].dropna()
        life = cal[(cal >= s.index.min()) & (cal <= s.index.max())]
        out[e] = s.reindex(s.index.union(life)).ffill(limit=FFILL_LIMIT).reindex(life)
    return pd.DataFrame(out).reindex(cal)


# ---------------------------------------------------------------------------- returns and signals
def held_expiry(month_end: pd.Timestamp, expiries) -> pd.Timestamp:
    """Nearest contract that does not expire before the end of the month."""
    later = [e for e in expiries if e > month_end]
    return min(later) if later else pd.NaT


def root_series(panel: pd.DataFrame, cal: pd.DatetimeIndex):
    """Daily return of the held contract, held expiry, and month-end carry."""
    p = on_calendar(panel, cal)
    expiries = list(p.columns)
    months = pd.Series(cal, index=cal).dt.to_period('M')
    held = months.map({m: held_expiry(m.to_timestamp(how='end').normalize(), expiries) for m in months.unique()})
    ret = pd.Series(np.nan, index=cal)
    for e in held.dropna().unique():
        days = held.index[held == e]
        s = p[e]
        r = 100 * (s / s.shift(1) - 1)
        ret.loc[days] = r.reindex(days)
    # fallback: if the held contract has no price on t-1 or t, the nearest unexpired contract that has both
    one_day = 100 * (p / p.shift(1) - 1)
    for t in ret.index[ret.isna()][1:]:
        alive = [e for e in expiries if e >= t and pd.notna(one_day.at[t, e])]
        if alive:
            ret.at[t] = one_day.at[t, min(alive)]
            held.at[t] = min(alive)
            FALLBACK_DAYS.append(t)
    # carry measured on the last trading day of M for the contract held in M+1
    last_days = pd.Series(cal, index=cal).groupby(months).max()
    carry = {}
    for m, d in last_days.items():
        nxt = held_expiry((m + 1).to_timestamp(how='end').normalize(), expiries)
        if pd.isna(nxt):
            continue
        after = [e for e in expiries if e > nxt]
        if not after:
            continue
        e2 = min(after)
        f1, f2 = p.at[d, nxt], p.at[d, e2]
        if pd.notna(f1) and pd.notna(f2) and f1 > 0 and f2 > 0:
            carry[m + 1] = 100 * (np.log(f1) - np.log(f2)) * 365 / (e2 - nxt).days
    return ret, held, pd.Series(carry)


def hml(returns: pd.DataFrame, signal: pd.DataFrame, assets) -> pd.Series:
    """Daily return of the equal-weighted top-minus-bottom tercile portfolio, re-sorted monthly."""
    months = returns.index.to_period('M')
    out = pd.Series(np.nan, index=returns.index)
    for m in months.unique():
        if m not in signal.index:
            continue
        sig = signal.loc[m, assets].dropna()
        k = len(sig) // 3
        if k == 0:
            continue
        ranked = sig.sort_values()
        lo, hi = ranked.index[:k], ranked.index[-k:]
        days = returns.index[months == m]
        out.loc[days] = returns.loc[days, hi].mean(axis=1) - returns.loc[days, lo].mean(axis=1)
    return out


# ---------------------------------------------------------------------------- macro block (as for gold and the industry portfolios)
def read_french(path: str, header_startswith: str) -> pd.DataFrame:
    with open(path, encoding='latin-1') as f:
        lines = f.read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.replace(' ', '').startswith(header_startswith))
    j = i + 1
    while j < len(lines) and lines[j].strip() and lines[j].strip()[0].isdigit():
        j += 1
    df = pd.read_csv(io.StringIO("\n".join(lines[i:j])))
    df = df.rename(columns={df.columns[0]: 'Date'})
    df.columns = [c.strip() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'].astype(str).str.strip(), format='%Y%m%d')
    return df.replace([-99.99, -999], np.nan).set_index('Date')


def read_fred(series: str) -> pd.Series:
    df = pd.read_csv(os.path.join(RAW, f'FRED_{series}.csv'), na_values=['.', ''])
    df.columns = ['Date', series]
    df['Date'] = pd.to_datetime(df['Date'])
    return df.set_index('Date')[series]


def read_yahoo(name: str) -> pd.Series:
    s = pd.read_csv(os.path.join(RAW, f'yahoo_{name}.csv'), parse_dates=['Date']).set_index('Date')['close']
    return s[s > 0]


def level_on(level: pd.Series, cal: pd.DatetimeIndex) -> pd.Series:
    return level.reindex(level.index.union(cal)).sort_index().ffill().reindex(cal)


def macro_block(cal: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=cal)
    out['d_tips'] = level_on(read_fred('DFII10'), cal).diff()
    out['d_be'] = level_on(read_fred('T10YIE'), cal).diff()
    out['dlog_dxy'] = 100 * np.log(level_on(read_yahoo('DX-Y_NYB'), cal)).diff()
    out['dlog_vix'] = 100 * np.log(level_on(read_yahoo('VIX'), cal)).diff()
    return out


# ---------------------------------------------------------------------------- main
def main():
    ff = read_french(os.path.join(RAW, 'F-F_Research_Data_5_Factors_2x3_daily.csv'), ',Mkt-RF')
    cal = ff.index[(ff.index >= WARMUP) & (ff.index <= END)]

    rets, helds, carries, report = {}, {}, {}, []
    for root in ROOTS:
        if root in CONTRACT_ROOTS:
            panel, (checked, bad) = panel_from_contracts(root), (0, 0)
        else:
            panel, (checked, bad) = panel_from_continuous(root)
        FALLBACK_DAYS.clear()
        rets[root], helds[root], carries[root] = root_series(panel, cal)
        report.append((root, panel.shape[1], checked, bad, len(FALLBACK_DAYS),
                       FALLBACK_DAYS[-1].date() if FALLBACK_DAYS else ''))
    returns = pd.DataFrame(rets)
    held = pd.DataFrame(helds)
    carry = pd.DataFrame(carries)

    # 12-month momentum at the end of M, used in M+1
    monthly = np.log1p(returns / 100).groupby(returns.index.to_period('M')).sum(min_count=1)
    mom_signal = monthly.rolling(12, min_periods=12).sum()
    mom_signal.index = mom_signal.index + 1

    factors = pd.DataFrame(index=cal)
    factors['mkt_rf'] = ff['Mkt-RF'].reindex(cal)
    factors['rf'] = ff['RF'].reindex(cal)
    factors = factors.join(macro_block(cal))
    factors['avg'] = returns.mean(axis=1)
    factors['carry'] = hml(returns, carry, ROOTS)
    factors['mom'] = hml(returns, mom_signal, ROOTS)
    for root in ROOTS:
        others = [r for r in ROOTS if r != root]
        factors[f'avg_{root}'] = returns[others].mean(axis=1)
        factors[f'carry_{root}'] = hml(returns, carry, others)
        factors[f'mom_{root}'] = hml(returns, mom_signal, others)

    keep = (cal >= START) & (cal <= END)
    returns, factors, held = returns[keep], factors[keep], held[keep]
    data = returns.add_prefix('ret_').join(factors)
    data.index.name = held.index.name = 'Date'
    data.to_csv(os.path.join(HERE, 'futures_data.csv'), float_format='%.6f')
    held.to_csv(os.path.join(HERE, 'futures_held.csv'))

    print("contracts per root, and inferred-vs-observed expiry checks (continuous RICs):")
    for root, n, checked, bad, nfb, last_fb in report:
        fb = f"   fallback days {nfb} (last {last_fb})" if nfb else ""
        print(f"   {root:4s} contracts {n:3d}   expiry checks {checked:6d}   mismatches {bad}{fb}")
    print(f"futures_data: {returns.index.min().date()} -> {returns.index.max().date()}, {len(returns)} days")
    print(f"missing returns per root: {returns.isna().sum().to_dict()}")
    print(f"missing factor values: {int(factors.isna().sum().sum())}")


if __name__ == '__main__':
    main()
