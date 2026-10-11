#!/usr/bin/env python
"""
Downloads the daily futures data of the commodity universe from LSEG Workspace:
- the first three continuous contracts (RIC <root>c1, c2, c3) of every root, and the fourth
  (<root>c4) for heating oil, RBOB gasoline and Brent: their contracts expire at a month end,
  so on the last trading day of a month the contract after the held one is the fourth
  (needed for the carry signal);
- the individual (expired) contracts of the roots whose continuous RICs carry no
  expiry dates before 2018-2025 (platinum, palladium, orange juice, oats, feeder
  cattle). Expired contracts use the RIC <root><month code><year digit>^<decade digit>;
  contracts expiring from 2024 on use two year digits (<root><month><yy>^<decade digit>).

Needs the Workspace desktop application to be running and signed in
(the API proxy listens on 127.0.0.1:9000). Each RIC is saved to
data/raw_lseg/<RIC>.csv; existing files are kept (delete a file to refetch it).

    python data/fetch_futures.py
"""
import os
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lseg_compat  # noqa: E402,F401  (refinitiv-data / httpx compatibility)
import refinitiv.data as rd  # noqa: E402

UNIVERSE = {
    "energy":    ["CL", "LCO", "NG", "HO", "RB", "LGO"],
    "metals":    ["GC", "SI", "HG", "PL", "PA"],
    "grains":    ["C", "W", "S", "BO", "SM", "KW", "O"],
    "softs":     ["KC", "SB", "CC", "CT", "OJ"],
    "livestock": ["LC", "LH", "FC"],
}
FIELDS = ["SETTLE", "TRDPRC_1", "SETL_PCHNG", "OPINT_1", "ACVOL_UNS", "EXPIR_DATE", "CRT_MNTH"]
START, END = 2008, 2025
OUT = os.path.join(HERE, "raw_lseg")
SLOTS = {"HO": 4, "RB": 4, "LCO": 4}         # continuous contracts per root (default 3)
# listed delivery months (exchange contract specifications)
CONTRACT_MONTHS = {"PL": "FJNV", "PA": "HMUZ", "OJ": "FHKNUX", "O": "HKNUZ", "FC": "FHJKQUVX"}
CONTRACT_YEARS = range(2008, 2027)          # contracts expiring 2008-2026
OUT_CONTRACTS = os.path.join(HERE, "raw_lseg", "contracts")


def fetch(ric):
    parts = []
    for y0 in range(START, END + 1, 3):
        y1 = min(y0 + 2, END)
        for attempt in range(3):
            try:
                d = rd.get_history(ric, fields=FIELDS, start=f"{y0}-01-01", end=f"{y1}-12-31")
                break
            except Exception as e:  # transient API errors: retry
                if attempt == 2:
                    raise RuntimeError(f"{ric} {y0}-{y1}: {e}") from e
                time.sleep(3)
        if d is not None and len(d):
            parts.append(d)
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "Date"
    return df


def contract_ric(root, month, year):
    digits = f"{year % 100:02d}" if year >= 2024 else f"{year % 10}"
    return f"{root}{month}{digits}^{(year // 10) % 10}"


def fetch_contracts():
    os.makedirs(OUT_CONTRACTS, exist_ok=True)
    missing = []
    for root, months in CONTRACT_MONTHS.items():
        for year in CONTRACT_YEARS:
            for m in months:
                ric = contract_ric(root, m, year)
                path = os.path.join(OUT_CONTRACTS, f"{root}_{year}{m}.csv")
                if os.path.exists(path):
                    continue
                try:
                    d = rd.get_history(ric, fields=["SETTLE", "TRDPRC_1", "OPINT_1", "EXPIR_DATE"],
                                       start=f"{year - 3}-01-01", end=f"{year}-12-31")
                except Exception:
                    d = None
                if d is None or not len(d.dropna(how="all")):
                    missing.append(ric)
                    continue
                d.index.name = "Date"
                d.to_csv(path)
        print(f"contracts {root}: done")
    if missing:
        print(f"{len(missing)} contracts returned no data:", " ".join(missing))


def main():
    os.makedirs(OUT, exist_ok=True)
    rd.open_session()
    print("session:", rd.session.get_default().open_state)
    for sector, roots in UNIVERSE.items():
        for root in roots:
            for n in range(1, SLOTS.get(root, 3) + 1):
                ric = f"{root}c{n}"
                path = os.path.join(OUT, f"{ric}.csv")
                if os.path.exists(path):
                    continue
                df = fetch(ric)
                df.to_csv(path)
                print(f"{sector:9s} {ric:6s} {len(df):5d} rows  {df.index.min().date()} .. {df.index.max().date()}")
    fetch_contracts()
    rd.close_session()


if __name__ == "__main__":
    main()
