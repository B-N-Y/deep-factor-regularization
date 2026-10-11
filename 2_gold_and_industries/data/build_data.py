#!/usr/bin/env python
"""
build_data.py
-------------
Builds the daily datasets of gold and the industry portfolios from public sources (downloaded once
into data/raw/; the CSVs written here are what the experiments read, no internet needed).

gold_factors.csv  (calendar = GLD trading days)
    gold_ret    GLD simple return, %                         Yahoo Finance (GLD)
    rf          one-month T-bill, daily simple rate, %       Ken French library
    mkt_rf      market excess return, %                      Ken French library
    d_tips      change in 10-year TIPS yield, pp             FRED DFII10
    d_be        change in 10-year breakeven inflation, pp    FRED T10YIE
    dlog_dxy    100 x log change, US dollar index            Yahoo (DX-Y.NYB)
    dlog_brent  100 x log change, Brent crude futures        Yahoo (BZ=F)
    dlog_vix    100 x log change, VIX                        Yahoo (^VIX)

industry_factors.csv  (calendar = CRSP trading days)
    10 value-weighted industry portfolio returns (NoDur ... Other), %;
    mkt_rf, smb, hml, rmw, cma, rf (Fama-French 5 factors, 2x3), mom; all %;
    d_tips, d_be, dlog_dxy, dlog_brent, dlog_vix: the macro block, defined as in
    gold_factors.csv but on the CRSP calendar.

Every factor change covers the same interval as the asset return (close t-1 to
close t): level series are carried forward onto the asset's trading calendar
before differencing, so a market holiday shows up as a zero change.

Usage:  python data/build_data.py            (uses files in data/raw, downloads Yahoo series)
"""
import io
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'raw')
START, END = '2005-01-01', '2026-01-01'


def read_french(path: str, header_startswith: str, section_title: str = None) -> pd.DataFrame:
    """Parse a Ken French daily CSV block (optionally the block after section_title)."""
    with open(path, encoding='latin-1') as f:
        lines = f.read().splitlines()
    i = 0
    if section_title:
        i = next(k for k, l in enumerate(lines) if section_title in l)
    i = next(k for k in range(i, len(lines)) if lines[k].replace(' ', '').startswith(header_startswith))
    j = i + 1
    while j < len(lines) and lines[j].strip() and lines[j].strip()[0].isdigit():
        j += 1
    df = pd.read_csv(io.StringIO("\n".join(lines[i:j])))
    df = df.rename(columns={df.columns[0]: 'Date'})
    df.columns = [c.strip() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'].astype(str).str.strip(), format='%Y%m%d')
    df = df.replace([-99.99, -999], np.nan)
    return df


def read_fred(series: str) -> pd.Series:
    df = pd.read_csv(os.path.join(RAW, f'FRED_{series}.csv'), na_values=['.', ''])
    df.columns = ['Date', series]
    df['Date'] = pd.to_datetime(df['Date'])
    return df.set_index('Date')[series]


def read_yahoo(ticker: str) -> pd.Series:
    """Close (adjusted where applicable) for ticker, cached in data/raw."""
    cache = os.path.join(RAW, f"yahoo_{ticker.replace('^', '').replace('=', '_').replace('.', '_')}.csv")
    if not os.path.isfile(cache):
        import yfinance as yf
        d = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        col = 'Adj Close' if 'Adj Close' in d.columns else 'Close'
        d[[col]].rename(columns={col: 'close'}).to_csv(cache)
    s = pd.read_csv(cache, parse_dates=['Date']).set_index('Date')['close']
    return s[s > 0]


def on_calendar(level: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Carry a level series forward onto the asset's trading calendar."""
    return level.reindex(level.index.union(calendar)).sort_index().ffill().reindex(calendar)


def macro_block(cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily changes of the macro series over the intervals of the given trading calendar."""
    out = pd.DataFrame(index=cal)
    out['d_tips'] = on_calendar(read_fred('DFII10'), cal).diff()
    out['d_be'] = on_calendar(read_fred('T10YIE'), cal).diff()
    for name, tk in [('dxy', 'DX-Y.NYB'), ('brent', 'BZ=F'), ('vix', '^VIX')]:
        out[f'dlog_{name}'] = 100 * np.log(on_calendar(read_yahoo(tk), cal)).diff()
    return out


def build_gold(ff5: pd.DataFrame) -> pd.DataFrame:
    gld = read_yahoo('GLD')
    cal = gld.index[(gld.index >= START) & (gld.index < END)]
    out = pd.DataFrame(index=cal)
    out['gold_ret'] = 100 * gld.reindex(cal).pct_change()
    ff = ff5.set_index('Date')
    out['rf'] = ff['RF'].reindex(cal)
    out['mkt_rf'] = ff['Mkt-RF'].reindex(cal)
    out = out.join(macro_block(cal))
    out = out.dropna()
    out.index.name = 'Date'
    return out


def build_industry(ff5: pd.DataFrame, mom: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    df = ind.merge(ff5, on='Date', how='inner').merge(mom, on='Date', how='inner')
    df = df.rename(columns={'Mkt-RF': 'mkt_rf', 'SMB': 'smb', 'HML': 'hml', 'RMW': 'rmw',
                            'CMA': 'cma', 'RF': 'rf', 'Mom': 'mom'})
    df = df[(df['Date'] >= START) & (df['Date'] < END)].set_index('Date')
    df = df.join(macro_block(df.index)).dropna()
    df.index.name = 'Date'
    return df


def main():
    ff5 = read_french(os.path.join(RAW, 'F-F_Research_Data_5_Factors_2x3_daily.csv'), ',Mkt-RF')
    mom = read_french(os.path.join(RAW, 'F-F_Momentum_Factor_daily.csv'), ',Mom')
    ind = read_french(os.path.join(RAW, '10_Industry_Portfolios_Daily.csv'), ',NoDur',
                      section_title='Average Value Weighted Returns -- Daily')

    gold = build_gold(ff5)
    industry = build_industry(ff5, mom, ind)
    gold.to_csv(os.path.join(HERE, 'gold_factors.csv'), float_format='%.6f')
    industry.to_csv(os.path.join(HERE, 'industry_factors.csv'), float_format='%.6f')
    for name, d in [('gold_factors', gold), ('industry_factors', industry)]:
        print(f"{name}: {d.index.min().date()} -> {d.index.max().date()}, {len(d)} rows, columns {list(d.columns)}")


if __name__ == '__main__':
    main()
