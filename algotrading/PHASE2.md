# Investment firm — SEC financial analyst implementation

**Scope:** $50,000 simulated portfolio, US stocks/ETFs, no leverage and no broker orders.

## Components

- `sec_analyst.py`: ticker/CIK lookup; SEC submissions and US-GAAP companyfacts; as-of 10-K/10-Q selection; filing-matched financial metrics and sourced JSON.
- `sec_narrative.py`: downloads full 10-K/10-Q HTML using declared SEC bot identity; extracts *candidate* MD&A and risk-factor passages, marked unverified; enumerates subsequent 8-K/8-K/A filings for review.
- `research_runner.py`: financial analyst + filing narrative + subsequent 8-K surveillance -> CIO/risk committee evidence packet and decision log.
- `firm.py`: fail-closed risk veto and trade-proposal gate. This is not an execution engine.
- `backtest.py`: simulator requiring truly out-of-sample predictions.
- `test_sec_analyst.py`, `test_sec_narrative.py` and existing tests: synthetic regression cases.

## Run locally

Install with `python -m pip install -r algotrading/requirements.txt`.
Declare a genuine SEC bot User-Agent with application name and contact email, e.g. `SEC_USER_AGENT="Research contact@example.com"` (on PowerShell `$env:SEC_USER_AGENT="Research contact@example.com"`).
Execute `python algotrading/research_runner.py AAPL --asof 2026-10-08`.
Execute `python -m pytest -q algotrading` to run tests.
Results are stored under `output/` unless `--out` is supplied.

## Interpretation / limitations

- Facts are matched to the **exact accession and filing date** known at as-of time, not latest amended historical values.
- 10-Q revenue/earnings use ~quarter duration; operating cash flow may be fiscal year-to-date; 10-K uses annual periods.
- Failing to fetch SEC data, missing financial metrics, or stale filings must never produce an approved order.
- Narrative passages and subsequently filed 8-K reports can now be collected, but the extraction is heuristic and **not a verified reading or interpretation** of those disclosures. Agent-review and accurate extraction of notes/8-K contents are pending.
- No market news agent, quantitative model execution, multi-asset portfolio optimizer, broker API, or autonomous execution is active.
- Even complete structured facts result in **HOLD** until financial narrative review, contemporaneous news, and validated quant predictions are implemented.
- The SEC publishes these APIs without keys. Declare the User-Agent and respect SEC fair-access rules.

## Next development milestones

1. Add source-grounded qualitative interpretation, verification of MD&A/risk-factor section boundaries, notes to accounts, and subsequent 8-K contents.
2. Build market-news intelligence with source provenance and as-of timestamp filters.
3. Convert the repository's LSTM/GRU/CNN/XGBoost predictions into leakage-free walk-forward returns.
4. Add quant and risk committee disagreement resolution; implement multi-asset paper simulation.
5. Integrate a broker sandbox with reconciliation and a kill switch, *after* validation.

Daily 1% net is a research target, not an expected or promised outcome.