# Phase 4 — Multi-strategy investment firm research laboratory

**Status: IMPLEMENTED as historical research; NOT live/paper-broker order placement.**

## User's pilot mandate

- US stocks and ETFs; initial capital **USD 50,000 (simulation only)**.
- No leverage, no short positions, max 10% per security, max 20% per equity sector, max 50% gross invested; remaining cash has zero assumed interest.
- Independent drawdown circuit breaker after a 12% peak-to-trough loss within the evaluated portfolio; stops new buys and liquidates at next available open in simulation.
- Typical rebalance every five sessions; immediate proposed liquidation when the observed market regime turns risk-off/high-volatility.
- Transaction cost assumption: **5bps commission + 5bps slippage per one-way traded dollar**. Sensitivity: 10bps commission + 20bps slippage.
- Historical signals observed at close *t* execute at *open t+1* and earn returns through *open t+2*. No same-day fills at the previous close.

## Working research modules

| Module | Scope |
|---|---|
| `portfolio_lab.py` | Aligned adjusted-price market panels, hashed input snapshots, as-of market regime, portfolio strategy proposals, concentration vetoes, actual weight drift, cash accounting, transaction costs, drawdown stop, SPY benchmarking, trade ledger |
| `portfolio_experiments.py` | End-to-end CLI: Yahoo OHLCV snapshot, 11 ETF sector rotation / sector pullback, 23 stock cross-sectional ranking, 504 out-of-sample-style evaluation sessions, sources, JSON summary and detailed CSVs |
| `bull_bear.py` | Independent deterministic research roles. Requires real linked evidence, abstains when missing; neither role authorizes an order |
| `earnings_events.py` | Point-in-time earnings event contract with exact availability timestamp, source URL and no lookahead; **no actual earnings strategy results until historical point-in-time EPS surprise data are sourced** |
| `paper_broker_readonly.py` | **GET-only** Alpaca PAPER account/clock readiness with immutable paper host, redacted account IDs, no order submission endpoint |
| `quant_agent.py` | Existing CPU forecast research and bull/bear memo handoff to CIO with fail-closed HOLD; this research-only agent is separate from the multi-asset strategy runner |
| `test_portfolio_lab.py`, `test_earnings_events.py`, `test_paper_broker_readonly.py` | Synthetic tests for no-future-info signals, exact next-open execution, shared cash portfolio, sector budget, circuit breaker, source timestamps and broker restrictions |

## Historical experiment execution

GitHub workflow: [multi-asset portfolio research](https://github.com/DanielAdam1992/Stocks/actions/workflows/portfolio-strategies.yml).
Original completed run: [October 8, 2026](https://github.com/DanielAdam1992/Stocks/actions/runs/37820464293).

Each of the following is a **504-session simulated portfolio total return, not a daily return**:

| Portfolio strategy | Initial results (10bps one-way cost) | Full-time 50% SPY/50% cash baseline |
|---|---:|---:|
| Sector rotation | +7.10% | +19.84% |
| Sector mean reversion | +0.35% | +19.84% |
| Cross-industry stock momentum | +13.52% | +19.84% |

All underperformed the passive risk-budget benchmark. They do not meet any institutional promotion standard, let alone sustained 1% net per day.

The full historical input snapshots, dates, turnover, trade journal, equity curve, stop events, and metrics are downloadable in the GitHub Actions artifact (retained for 30 days).

**These numbers are preliminary.** The relative positions of strategies may shift with changing data adjustments, source revisions, different inception points, universe definitions, revised fees and risk parameter choices. We explicitly disclose survivorship bias because the fixed representative 23-stock universe is retrospectively selected.

## How to reproduce

Install market dependencies:

```bash
python -m pip install -r algotrading/requirements-market.txt
python -m pytest -q algotrading
python algotrading/portfolio_experiments.py --asof 2026-10-07 --start 2018-01-01 --test-sessions 504 --rebalance 5 --fee-bps 5 --slippage-bps 5
```

For higher costs using **exactly the original price files**:

```bash
python algotrading/portfolio_experiments.py --data-dir output/portfolio_experiments/market_inputs --asof 2026-10-07 --test-sessions 504 --rebalance 5 --fee-bps 10 --slippage-bps 20 --out output/portfolio_cost_stress
```

Use snapshots for reproducibility: repeated fresh data downloads can be revised. Returns are modeled with next-open *adjusted* OHLC, not an actual broker opening auction, and omit taxes, financing, borrow, dividends timing, exchange fees, and impact beyond fixed slippage assumptions.

## Paper brokerage connection — read-only boundary

Alpaca docs confirm the PAPER endpoint is `https://paper-api.alpaca.markets`, including `GET /v2/account` and order APIs. The current adapter intentionally implements only GET health checks, NOT order submissions.

After an Alpaca paper account has been established, keep credentials out of GitHub and set locally:

```bash
export APCA_API_KEY_ID="YOUR_PAPER_KEY"
export APCA_API_SECRET_KEY="YOUR_PAPER_SECRET"
python algotrading/paper_broker_readonly.py --check
```

PowerShell: `$env:APCA_API_KEY_ID="..."` and `$env:APCA_API_SECRET_KEY="..."`.

Broker keys have **not** been connected by this work. `broker_orders` are always empty in investment committee research. Paper account balance is not represented as the $50,000 simulated benchmark until separately verified.

## Investment agent roles & limits

- Quant researcher: time-aware LSTM/GRU/CNN/XGBoost and momentum signals. Existing checkpoint weights are not assumed valid.
- Fundamental researcher: SEC 10-K, 10-Q and 8-K ingestion with evidence; narrative extraction candidates still require verification.
- Market intelligence: regime metrics operational; licensed/live headline analysis **not yet operational**.
- Bull and bear researchers: deterministic adversarial memo structure operational, not independent agentic LLM workers yet.
- CIO / portfolio manager: constructs research targets, never brokers a trade.
- Independent risk manager: rejects shorts, leverage, missing sector maps, position/sector/gross excess and large portfolio drawdowns.
- Broker operations: read-only paper-health connector, no credentials or order API enabled.

## Next blockers before autonomous paper execution

1. Source-stamped historical earnings event feed, backtesting on unseen regimes and independent research validation.
2. Full cross-sectional/sector portfolio attribution, provider-adjustment reconciliation, delisting and survivorship-bias-safe universe.
3. Realistic limit/market orders, partial fills, market-hours controls, idempotent order IDs, balance reconciliation and fail-safe recovery.
4. A paper account with explicit keys, full order and position audit logs, and a documented human-authorized activation process.
5. A statistically defensible strategy that beats its risk-matched passive comparator after all costs across multiple independent out-of-sample periods.

No real-money or paper-broker orders are authorized by the current modules.
