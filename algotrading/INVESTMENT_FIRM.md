# Autonomous investment firm — architecture v0.1

**Mandate:** US stocks and ETFs; $50,000 paper portfolio; no leverage; no live orders.

## Roles and responsibility boundaries

| Role | Inputs | Output | Restrictions |
|---|---|---|---|
| Financial filings analyst | SEC EDGAR 10-K/10-Q/8-K, earnings releases | Cited filing facts, dates, accounting risks | No invented figures; verify filing periods |
| Market intelligence | News, rates, macro releases, corporate actions | Time-stamped event memo | No trading based on unverified headlines |
| Quant researcher | Existing XGBoost, LSTM/GRU/CNN and fastai models | Out-of-sample expected return, calibration, uncertainty | No training/test leakage |
| Investment strategist | Quant + fundamentals + catalysts | Thesis, scenarios, entry/exit proposal | Must disclose disagreements |
| Portfolio manager | Holdings, cash, exposures, correlations | Target allocations | Cash, concentration and sector limits |
| Independent risk officer | Volatility, drawdowns, liquidity, limits | Approve/veto with reason | Cannot be overridden by CIO |
| CIO | All signed research packets | Decision record | Cannot place orders |
| Execution agent | Approved orders, broker state | Paper orders, fill and rejection log | Idempotency, market-hours and kill-switch checks |
| Compliance/audit | Sources, approvals, orders, fills | Immutable decision trail | Fail closed on missing provenance |

## Data contracts
Each research packet needs ticker, as-of timestamp, source URLs, filing period, extracted metrics, quantitative prediction horizon, model version, confidence/calibration, and any red flags. Never mark `financials_reviewed` or `news_reviewed` true until evidence is actually retrieved and checked.

The current `firm.py` is **only a deterministic committee gate**: missing evidence or risk flags result in HOLD; successful review yields PROPOSE_BUY, not an order. It does not read filings, invoke LLMs, or execute trades.

## Implementation sequence
1. SEC EDGAR filing ingestion with ticker/CIK mapping, filing dates, structured companyfacts and source citations.
2. Financial statement metrics and source-grounded analyst memo; tests for missing/stale filings.
3. Chronological out-of-sample ML inference with calibrated forecast horizon and leakage-free features.
4. Committee orchestration with typed inputs, agent disagreement and risk veto.
5. Multi-asset portfolio accounting and next-session execution backtest, including slippage/fees.
6. Broker sandbox, reconciliation, alerting, circuit breaker and operational audit.
7. Paper forward testing and independent review before any consideration of live trading.

A 1% net daily profit goal is not a reliable assumption; optimize risk-adjusted performance and survival rather than promising returns.

## Current implementation checkpoint (2026-10-08)

- **Operational code (research):** SEC financial-facts analyst; filing HTML candidate-excerpt reader; SEC 8-K current-report listing; committee runner and deterministic veto gate; standalone backtest harness.
- **Not operational:** true LLM-based narrative or news analysis, saved-model inference from existing Colab notebooks, live market intelligence, trade portfolio optimizer, autonomous broker connection, compliance automation, and actual model profitability validation.
- Every agent state is recorded in the research runner's JSON output. Unverified document excerpts and missing analyst roles **must not** be treated as approval.
- Draft PR #1 on a feature branch is for review; no automatic merge or live deployment.
