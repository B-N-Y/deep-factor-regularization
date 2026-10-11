#!/usr/bin/env python
"""
download_public_data.py
-----------------------
Step 1: downloads the public data of both samples into
1_commodity_futures/data/raw/ and 2_gold_and_industries/data/raw/.

    Kenneth R. French Data Library   Fama-French five factors (2x3, daily), momentum (daily),
                                     ten industry portfolios (daily)
    FRED                             DFII10 (10-year TIPS yield), T10YIE (10-year breakeven inflation)
    Yahoo Finance (yfinance)         DX-Y.NYB (US dollar index), ^VIX, GLD, BZ=F (Brent)

The commodity futures prices come from LSEG Workspace and are licensed; they are
downloaded with 1_commodity_futures/data/fetch_futures.py.

    python download_public_data.py
"""
import io
import os
import shutil
import urllib.request
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FUT_RAW = os.path.join(HERE, '1_commodity_futures', 'data', 'raw')
EQ_RAW = os.path.join(HERE, '2_gold_and_industries', 'data', 'raw')

FRENCH = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
FRENCH_FILES = {'F-F_Research_Data_5_Factors_2x3_daily_CSV.zip': 'F-F_Research_Data_5_Factors_2x3_daily.csv',
                'F-F_Momentum_Factor_daily_CSV.zip': 'F-F_Momentum_Factor_daily.csv',
                '10_Industry_Portfolios_daily_CSV.zip': '10_Industry_Portfolios_Daily.csv'}
FRED = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id='
FRED_SERIES = ['DFII10', 'T10YIE']
YAHOO = ['DX-Y.NYB', '^VIX', 'GLD', 'BZ=F']
START, END = '2005-01-01', '2026-01-01'
# files read by the futures pipeline (the gold and industry pipeline reads all of them)
FUTURES_NEEDS = ['F-F_Research_Data_5_Factors_2x3_daily.csv', 'FRED_DFII10.csv', 'FRED_T10YIE.csv',
                 'yahoo_DX-Y_NYB.csv', 'yahoo_VIX.csv']


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def yahoo_name(ticker: str) -> str:
    """Cache name used by 2_gold_and_industries/data/build_data.py."""
    return f"yahoo_{ticker.replace('^', '').replace('=', '_').replace('.', '_')}.csv"


def main():
    os.makedirs(FUT_RAW, exist_ok=True)
    os.makedirs(EQ_RAW, exist_ok=True)
    for zname, cname in FRENCH_FILES.items():
        with zipfile.ZipFile(io.BytesIO(fetch(FRENCH + zname))) as z:
            with open(os.path.join(EQ_RAW, cname), 'wb') as f:
                f.write(z.read(cname))
        print(f"French data library: {cname}")
    for series in FRED_SERIES:
        with open(os.path.join(EQ_RAW, f'FRED_{series}.csv'), 'wb') as f:
            f.write(fetch(FRED + series))
        print(f"FRED: {series}")
    import yfinance as yf
    for ticker in YAHOO:
        d = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        col = 'Adj Close' if 'Adj Close' in d.columns else 'Close'
        d.index.name = 'Date'
        d[[col]].rename(columns={col: 'close'}).to_csv(os.path.join(EQ_RAW, yahoo_name(ticker)))
        print(f"Yahoo Finance: {ticker}")
    for name in FUTURES_NEEDS:
        shutil.copy(os.path.join(EQ_RAW, name), os.path.join(FUT_RAW, name))
    print(f"files written to {os.path.relpath(EQ_RAW, HERE)} and {os.path.relpath(FUT_RAW, HERE)}")


if __name__ == '__main__':
    main()
