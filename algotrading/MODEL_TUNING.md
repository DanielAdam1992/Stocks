# Quant research and model tuning — Phase 3

**Mandate:** $50,000 USD paper portfolio, US equities/ETFs, no leverage, **no broker orders**.

## Why the original notebooks cannot be accepted unchanged

- `dataprep_for_train.ipynb` fits MinMaxScaler across the full supplied time series before selecting model features and training, allowing test-period distribution information into scaling.
- Same notebook uses `.fillna(method='bfill')` and sets the unknown last `y_next` equal to current `y`; neither is a leakage-safe supervised return label.
- Price forecast `R²` / in-sample image classifier accuracy do **not** establish profitable trading signals.
- Historical CSVs contain engineered/scaled data rather than complete adjusted OHLCV, so new, explicit, as-of market prices are used for executable-return research.
- Legacy serialized TensorFlow/fastai model artifacts are not established in the current GitHub tree; do not claim retraining preserved the exact original weights.

## Current evaluated CPU models

- Zero-return dummy forecaster (a difficult but essential baseline).
- Ridge(alpha=10,100) with train-only StandardScaler.
- HistGradientBoostingRegressor: restricted leaf count 7/15, min_samples_leaf 35/45, shrinkage and L2 penalty.
- Optional XGBoost depth 2/3 with regularization, low learning rate and controlled ensemble sizes.
- Temporal inner train/validation models selected via validation RMSE. Outer chronological walk-forward blocks are never used for hyperparameter selection.
- Purge/lag: a signal computed after close t executes at the following open t+1, holding until open t+2; at that moment training excludes the unresolved preceding label. Features are computed exclusively from observations at/before t.
- No global price scaling, no label/backfill imputation and no random test shuffle.

## Current market studies

- **Sector ETFs:** XLK/XLF/XLE/XLV/XLI/XLY/XLP/XLU/XLB/XLRE/XLC plus SPY index benchmark. See GitHub workflow `sector-quant.yml`.
- **Industry stocks:** 23 representative companies from all eleven sectors; tests company-specific industries rather than substituting the ETF for an entire sector. See workflow `industry-quant.yml`.
- **Post-selection robustness:** longer overlapping 252/504/756-session windows and 10/20/40 basis-point one-way fees/impact. This is explicitly not a new independent holdout or license to trade. See workflow `quant-stress.yml`.
- **Legacy architecture families:** optional scratch-retrained LSTM/GRU/causal CNN, with 22/66-session lookback, train-only normalization, early stopping and time-aware validation. Implemented in `neural_tuning.py`; depends on TensorFlow CPU and *does not* load the legacy saved weights. See `neural-quant.yml`.

## Reproduce

1. `python -m pip install -r algotrading/requirements-market.txt`
2. `python algotrading/sector_runner.py --xgboost --oos 252 --retrain 63 --out output/sector_study`
3. `python algotrading/sector_runner.py --tickers AAPL MSFT NVDA JPM V XOM SLB JNJ UNH CAT GE AMZN HD PG COST NEE DUK LIN FCX PLD AMT META GOOGL --xgboost --out output/industry_study`
4. `python algotrading/stress_runner.py --out output/stress_study`
5. To test neural nets separately (larger CPU dependency): `python -m pip install -r algotrading/requirements-neural.txt` then `python algotrading/neural_tuning.py AAPL --oos 126 --retrain 126 --epochs 12`.
6. `python -m pytest -q algotrading` runs synthetic safety and chronology regression tests.

All test results are produced on a GitHub Actions runner from data fetched on that run. Source prices can be adjusted or revised by the provider; prefer snapshot hashes and an institutional market-data vendor before live trading.

## Cost and data caveats

- `performance()` allocates at most 10% of the $50,000 portfolio to the studied instrument, with the remainder in non-interest-bearing cash; comparisons to 10%-allocated buy/hold and 100% buy/hold are labeled separately.
- Trade P&L is based on following adjusted open-to-open prices; no actual opening fills are simulated. Expected commission/impact is a simple one-way cost per change in weight.
- Backtested **net** excludes tax, financing, FX, vendor fees and end-of-test liquidation costs; it is not realized profit.
- Profitability screen is a *research shortlist* (positive Sharpe; >=200 OOS sessions; >=10 entries; lower MAE than predicting zero; beating same-allocated hold). It is **not** investment committee approval.
- Industry stocks and candidates were chosen by design, and selecting the strongest results creates multiple-testing bias. Larger OOS, deflated Sharpe and forward paper trading are required.
- 1% net profit/day is a **stretch hypothesis**, not an expected or guaranteed return.

## Research sources

- scikit-learn time-series validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- scikit-learn nested validation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- scikit-learn randomized search: https://scikit-learn.org/stable/modules/grid_search.html
- Deflated Sharpe ratio / selection bias: https://doi.org/10.3905/jpm.2014.40.5.094
- Sector ETF classifications from Select Sector SPDR: https://www.sectorspdrs.com/

## Next phase

1. Compare neural networks against CPU baselines **on the same forecast dates** and with equal computational/search budgets; do not cherry-pick across sectors.
2. Add purged validation for longer holding periods, embargoes, and an untouched final evaluation period; multiple-testing corrected risk metrics.
3. Build an asset-level, multi-position portfolio simulator with actual broker-calibrated commissions, opening auction slippage, cash and risk budget.
4. Connect quant research and SEC/news agent evidence to the CIO committee, then a broker paper account under independent risk controls.