"""Run multi-asset institution-style historical experiments, NEVER live orders.

Example: python algotrading/portfolio_experiments.py --asof 2026-10-07
Downloads adjusted OHLCV and saves input snapshots with an audit hash.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import pandas as pd

from portfolio_lab import (PortfolioConfig, prepare_market_panel,
                           propose_allocations, regime_frame, run_portfolio)
from sector_runner import BENCHMARK, INDUSTRY_STOCKS, SECTOR_ETFS, load_prices


def _build_universe(tickers: list[str], data_dir: str|None, start: str,
                    asof: str, out: Path) -> dict:
    prices={}
    price_dir=out/"market_inputs"
    price_dir.mkdir(parents=True,exist_ok=True)
    for ticker in tickers:
        df=load_prices(ticker,source_dir=data_dir,start=start)
        df=df.copy()
        df["Date"]=pd.to_datetime(df["Date"],errors="raise",utc=True).dt.tz_localize(None)
        df=df[df["Date"]<=pd.Timestamp(asof)].sort_values("Date")
        if df.empty:
            raise ValueError(f"No historical market observations for {ticker} by {asof}")
        prices[ticker]=df
        df.to_csv(price_dir/f"{ticker}.csv",index=False,float_format="%.10g")
        print(f"{ticker}: {len(df)} observations, latest={df['Date'].max().date()}",flush=True)
    return prices


def run_study(prices: dict, universe: str,
              cfg: PortfolioConfig=PortfolioConfig()) -> tuple[dict,dict]:
    """Evaluate all deterministic proposals; selection of best is NOT a trading decision."""
    if universe not in {"sector","industry"}:
        raise ValueError("Unknown experiment universe")
    panel=prepare_market_panel(prices)
    members=[s for s in panel.close.columns if s!="SPY"]
    if universe=="sector":
        sector_map={s:s for s in members}
        strategies=("sector_rotation","sector_mean_reversion")
    else:
        sector_map={s:INDUSTRY_STOCKS[s].split(" /",1)[0] for s in members}
        strategies=("cross_sectional_momentum",)
    states=regime_frame(panel.close["SPY"],cfg)
    regime_counts=states["regime"].tail(cfg.test_sessions).value_counts().to_dict()
    reports={}
    raw_outputs={}
    for name in strategies:
        weights,decisions=propose_allocations(panel,name,cfg,sector_map)
        curve,trades,summary=run_portfolio(panel,weights,cfg,sector_map)
        reports[name]={
            **summary,
            "strategy":name,"universe":universe,
            "universe_symbols":sorted(members),
            "market_regimes_last_test_sessions":regime_counts,
            "n_strategy_decisions":len(decisions),
            "description":"Fixed predeclared research hypotheses, not optimized on outer test",
            "eligible_to_trade":False,
            "reason_no_trades":"Backtesting only; no approved edge or broker integration",
        }
        raw_outputs[name]={"curve":curve,"trades":trades,"decisions":decisions}
        print(f"{universe}/{name} historical portfolio={summary['net_return_pct']:+.2f}% "
              f"passive risk-budget SPY={summary['spy_risk_budget_return_pct']:+.2f}% "
              f"maxdrawdown={summary['max_drawdown_pct']:.2f}% orders={summary['order_events']}",
              flush=True)
    return reports,raw_outputs


def main():
    p=argparse.ArgumentParser(description="Multi-strategy US equity portfolio research only")
    p.add_argument("--universe",choices=["all","sector","industry"],default="all")
    p.add_argument("--start",default="2018-01-01")
    p.add_argument("--asof",default=None,help="Last allowed US market observation YYYY-MM-DD")
    p.add_argument("--data-dir",default=None)
    p.add_argument("--test-sessions",type=int,default=504)
    p.add_argument("--rebalance",type=int,default=5)
    p.add_argument("--fee-bps",type=float,default=5)
    p.add_argument("--slippage-bps",type=float,default=5)
    p.add_argument("--out",default="output/portfolio_experiments")
    a=p.parse_args()
    asof=a.asof or datetime.now(timezone.utc).date().isoformat()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}",asof):
        p.error("Invalid --asof YYYY-MM-DD")
    if pd.Timestamp(asof).date()>datetime.now(timezone.utc).date():
        p.error("No future market information permitted")
    cfg=PortfolioConfig(test_sessions=a.test_sessions,rebalance_every=a.rebalance,
                        one_way_fee_bps=a.fee_bps,
                        one_way_slippage_bps=a.slippage_bps)
    cfg.validate()
    base=Path(a.out)
    base.mkdir(parents=True,exist_ok=True)
    result={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "paper_research_only":True,
        "input_asof":asof,
        "data_source":"user_csv" if a.data_dir else "yfinance_adjusted_ohlcv",
        "commission_slippage_bps_per_one_way_dollar":
            cfg.one_way_fee_bps+cfg.one_way_slippage_bps,
        "warnings":[
            "No backtest outcome is a live execution authorization",
            "Fixed representative industries suffer survivorship/selection bias",
            "Price data are vendor-adjusted and could later be revised",
            "Open-to-open estimated fills cannot establish broker-realized returns",
            "Hypothesis screening of multiple strategies requires independent forward confirmation",
        ],
        "runs":{},
        "config":asdict(cfg),
    }
    symbols=["SPY"]
    if a.universe in ("all","sector"):
        symbols.extend(SECTOR_ETFS)
    if a.universe in ("all","industry"):
        symbols.extend(INDUSTRY_STOCKS)
    prices=_build_universe(symbols,a.data_dir,a.start,asof,base)
    for name in ("sector","industry"):
        if a.universe not in ("all",name):
            continue
        subset={s:prices[s] for s in (["SPY"]+
               (list(SECTOR_ETFS) if name=="sector" else list(INDUSTRY_STOCKS)))}
        reports,raw=run_study(subset,name,cfg)
        result["runs"][name]=reports
        for strategy,artifact in raw.items():
            directory=base/name/strategy
            directory.mkdir(parents=True,exist_ok=True)
            for name2,table in artifact.items():
                table.to_csv(directory/f"{name2}.csv",index=False)
            (directory/"metrics.json").write_text(
                json.dumps(reports[strategy],indent=2,default=str),encoding="utf-8")
    (base/"summary.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print("Research saved to",base/"summary.json",flush=True)


if __name__=="__main__":
    main()
