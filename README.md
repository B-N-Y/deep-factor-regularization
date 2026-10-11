# How much regularization do deep factor models need? Evidence from commodity futures and industry portfolios

Code for the paper by Berna N. Yilmaz, Bilgi Yilmaz and Savaş Takan.

The paper estimates contemporaneous factor models in which the link between daily excess
returns and same-day factors is an LSTM, a GRU or a feed-forward network (MLP), and varies only
the regularization: no regularization, standard dropout, Monte Carlo dropout, three dropout
schemes with a consistency penalty (fixed, dynamic and adaptive rate, each with 14 divergences),
weight decay and early stopping. The benchmark is the linear factor regression on the same
inputs. The main sample is 26 commodity futures; gold and the ten Fama–French industry
portfolios form a second asset class.

## Contents

```
download_public_data.py         step 1: public data for both samples
requirements.txt
1_commodity_futures/            main sample: 26 commodity futures
    data/fetch_futures.py       LSEG Workspace download (licensed data)
    data/lseg_compat.py         compatibility shim used by fetch_futures.py
    data/build_futures.py       contract roll, daily returns, carry and momentum signals,
                                leave-one-out commodity factors, macro block
    data/validate_returns.py    check of the returns against LSEG's settlement changes
    src/                        data path, networks, regularization regimes, divergences,
                                training, evaluation
    run_grid_search.py          hyperparameter search for one test asset and network
    merge_results.py            combines a search that was split over several jobs
    run_seed_sensitivity.py     repeats the search with other seeds
    run_expanding_cv.py         expanding-window evaluation
    tests/test_pipeline.py      checks to run before any experiment
2_gold_and_industries/          second asset class: gold (GLD) and ten industry portfolios
    data/build_data.py          returns and factors from the public data
    src/, run_*.py, tests/      as above
    run_timing.py               computational cost per configuration
3_analysis/                     tables, figures and every number in the paper
    make_tables.py              reads the result folders of both samples
    loading_instability.py      instability of the linear factor loadings (H3)
    h3_benchmarks.py            ridge and rolling regressions, Chow test, tracking share (H3)
    divergence_scale.py         size and local order of the 14 consistency penalties
    requirements.txt            package versions used for the analysis
```

The modelling code (`src/models.py`, `src/training.py`, `src/divergences.py`,
`src/evaluation.py`, `src/utils.py`) and the `run_*.py` scripts are identical in the two folders;
`src/config.py` (test assets, factors) and `src/data.py` (data path) differ. Apart from comments
and docstrings, the files are the versions that produced the results in the paper.

## Steps

**1. Environment.** Python 3.9 and the packages in `requirements.txt` (the versions used for every
run in the paper; the runs used NVIDIA H100 and H200 GPUs, CPU also works).

```bash
pip install -r requirements.txt
```

**2. Public data** (Kenneth R. French Data Library, FRED, Yahoo Finance) for both samples:

```bash
python download_public_data.py
```

**3. Commodity futures data.** The settlement prices are licensed from LSEG and are not part of
the repository. With LSEG Workspace running and signed in:

```bash
cd 1_commodity_futures
python data/fetch_futures.py        # raw LSEG files -> data/raw_lseg/
python data/build_futures.py        # -> data/futures_data.csv, data/futures_held.csv
python data/validate_returns.py     # returns vs LSEG settlement changes
```

**4. Gold and industry portfolio data.**

```bash
cd 2_gold_and_industries
python data/build_data.py           # -> data/gold_factors.csv, data/industry_factors.csv
```

**5. Tests.** In each sample folder:

```bash
python tests/test_pipeline.py
```

**6. Experiments.** Run inside the sample folder. `--config` is `<asset>_<period>` (for example
`c_2010_2025` for corn, `ind_hlth_2010_2025` for the health portfolio; see `src/config.py`) and
`--model` is `lstm`, `gru` or `mlp`.

| Experiment in the paper | Command |
|---|---|
| Main search: all dropout regimes, 14 divergences, 50 configurations each | `python run_grid_search.py --config c_2010_2025 --model lstm --dropout none,monte_carlo,standard,fixed,dynamic,adaptive` |
| Weight decay and early stopping | `python run_grid_search.py --config c_2010_2025 --model lstm --dropout weight_decay,early_stopping --out-root results_altreg` |
| Log-scale penalty weight | `python run_grid_search.py --config c_2010_2025 --model lstm --dropout fixed,dynamic,adaptive --alpha-scale log --out-root results_logalpha` |
| Subperiods | the main search with `--config c_2010_2015`, `c_2015_2020`, `c_2020_2025` |
| Seeds (0, 7, 42, 123, 2026) | `python run_seed_sensitivity.py --config c_2010_2025 --model lstm --seed 7` (after the main search) |
| Expanding window | `python run_expanding_cv.py --config c_2010_2025 --model lstm --n-folds 5` (after the main search) |
| Computational cost (gold and industries) | `python run_timing.py --reps 10` |

The main search covers all 26 futures (and gold and the ten industry portfolios) with the three
networks. Subperiods and seeds use four representative assets per sample: gold, WTI, corn and
live cattle (`gc`, `cl`, `c`, `lc`) for the futures; gold, energy, high technology and
utilities (`gold`, `ind_enrgy`, `ind_hitec`, `ind_utils`) for the second sample. A search can be
split over several jobs with `--part <name>` and a subset of `--dropout`, and combined with
`python merge_results.py --config ... --model ... [--root ...]`.

**7. Tables, figures and numbers.** After the experiments of step 6 for both samples, from the
repository root:

```bash
pip install -r 3_analysis/requirements.txt
python 3_analysis/make_tables.py --out output
```

This writes `output/numbers.tex` (every number quoted in the paper, as LaTeX macros),
`output/tables/`, `output/figures/` and `output/MANIFEST.json`. The script also rebuilds the
futures data and repeats the validation of step 3, so it needs the LSEG downloads. Applied to the
result folders behind the paper, it reproduces its numbers, tables and figures exactly. The
analysis was run with Python 3.12 and the versions in `3_analysis/requirements.txt`; other SciPy
versions can give slightly different p-values.

## Outputs

Each search writes `results/<model>/<config>/` (or the folder given by `--out-root`):
`best_per_combo.csv` (one row per regime and divergence: selected configuration and its
training, validation and test fit), `all_trials.csv`, `benchmarks.csv` (linear factor
regression), `predictions.json` (validation and test fits), `selection.json` (selected
dropout model, chosen on the validation block only), `training_history.json` and `run_info.json`
(software versions, GPU, data split, seeds and checksums of the code).

## Data

Sample: January 2010 to December 2025; earlier data are used for the momentum signal and the
first input windows. Every series is split chronologically into training (4 January 2010 to
7 June 2021), validation (8 June 2021 to 18 September 2023) and test (19 September 2023 to
31 December 2025) blocks. The test block is not used for any choice.

MD5 checksums of the datasets used in the paper, for checking a rebuild:

| File | MD5 |
|---|---|
| `1_commodity_futures/data/futures_data.csv` | `fd1f357955e537df0557651a41ea22b8` |
| `1_commodity_futures/data/futures_held.csv` | `242069247ce6e5d56bb6f0b3b9561618` |
| `2_gold_and_industries/data/gold_factors.csv` | `c1188c5cac19f9a1a47feb61f80be1eb` |
| `2_gold_and_industries/data/industry_factors.csv` | `a2eb668b6b436cab20f2ee4229476d85` |

The public sources are revised from time to time (for example, the Fama–French factors are
updated annually), so a later download may differ slightly from the data used in the paper.

## License

The code is released under the MIT License (`LICENSE`). The commodity futures prices are
licensed from LSEG and are not included.
