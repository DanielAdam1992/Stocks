# Algo-trading research — Phase 1

**Mode:** paper/research only. **Starting portfolio:** $50,000 USD. **Universe:** US equities/ETFs. **No leverage.**

The existing notebooks train price forecasters and a separate chart-image classifier. Their headline metrics do not demonstrate tradable out-of-sample returns. Existing serialized models may not be checked into GitHub.

## Initial audit findings
- `dataprep_for_train.ipynb` sets the final unknown `y_next` to the current value; it also backfills missing feature values. Backfill can leak future information into training rows. Replace with train-only preprocessing and drop unknown targets.
- `predict_future.ipynb` recursively updates the first input feature while leaving other lagged/rolling features stale; its multi-day predictions require separate validation.
- Forecasting defaults to 7 days, while fastai labels use a 22-day horizon with +/-10% thresholds. These cannot be treated as directly interchangeable.
- Existing notebooks depend on Google Colab paths and serialized model artifacts. Need reproducible local model inference before integrating them.

## Backtesting harness
`backtest.py` accepts a CSV with `Date,Close,predicted_return`. **Predictions must be generated walk-forward, out of sample, as of the close of that date**; never use fitted predictions from the training set or forecasts generated with later data. The harness takes a long-or-cash position at the next daily close-to-close interval, applies a configurable turnover fee, and compares buy-and-hold.

```bash
pip install pandas numpy pytest
python algotrading/backtest.py path/to/oos_predictions.csv --output results.csv
pytest -q algotrading/test_backtest.py
```

Default research assumptions: 25% maximum asset exposure, 0.5% predicted-return entry threshold, 5 basis points per unit of turnover. These are placeholders, **not optimized or validated**. The close-to-close simulation is an approximation, not executable fill modeling; it excludes bid/ask spread, borrow, market impact, taxes and FX. Do not use for live trading.

## Next step
Extract model inference and train-only scalers into standalone modules. Generate strictly chronological walk-forward predictions for AAPL/MSFT/NVDA; run this harness and compare against buy-and-hold and no-skill baselines. Add realistic next-open execution, slippage, multi-asset portfolio accounting and stress tests before broker paper-trading integration.
