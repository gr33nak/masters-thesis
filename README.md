# Beyond Chain-Ladder: Claims Reserving in the Machine Learning Era

Code accompanying the master's thesis. The repository provides the benchmarking
notebook and the supporting Python package used to compare classical, machine
learning, and deep learning methods for non-life claims reserving on synthetic
micro-data.

## Contents

- [`notebook_analysis.ipynb`](notebook_analysis.ipynb) —
  end-to-end notebook that loads the synthetic dataset, runs all reserving methods,
  performs rolling-origin backtests, computes bootstrap uncertainty, and reproduces
  the figures and tables reported in the thesis.
- [`simulated_data.RData`](simulated_data.RData) —
  synthetic claim-level dataset used throughout the notebook, generated with the
  simulation machine of Gabrielli & Wüthrich (2019). Loaded in Python via `pyreadr`.
- [`python_reserving/`](python_reserving/) — Python package with the reusable
  building blocks called from the notebook:
  - [`benchmark_engine.py`](python_reserving/benchmark_engine.py) — Chain-Ladder,
    GLM-ODP, Cape Cod, and bootstrap uncertainty on aggregate triangles.
  - [`synthetic_loader.py`](python_reserving/synthetic_loader.py) — loaders for
    the simulation-machine output and helpers to build paid triangles and
    true-ultimate references.
  - [`rolling_origin.py`](python_reserving/rolling_origin.py) — rolling-origin and
    static benchmark drivers across valuation cut-offs.
  - [`ml_pipeline.py`](python_reserving/ml_pipeline.py) — claim-level XGBoost
    pipeline with snapshot construction and leakage controls.
  - [`dl_pipeline.py`](python_reserving/dl_pipeline.py) — claim-level deep
    learning rolling-origin backtest.
  - [`dl_triangle.py`](python_reserving/dl_triangle.py) — triangle-level deep
    learning backtest and comparison utilities.

## Requirements

Python 3.10+ with:

```
pandas numpy statsmodels scikit-learn xgboost tensorflow pyreadr jupyter
```

## Usage

```bash
pip install pandas numpy statsmodels scikit-learn xgboost tensorflow pyreadr jupyter
jupyter notebook notebooks/benchmark_engine_draft.ipynb
```

## Citation

If you use this code, please cite the thesis:

> Zelený, O. *Beyond Chain-Ladder: Claims Reserving in the Machine Learning Era.*
> Master's thesis, Charles University, 2026.
