"""Compare walk-forward forecast and simulated trading results across US sector ETFs.

Sector ETFs proxy industries; this is NOT stock-by-stock industry classification.
Install yfinance for downloading historical, adjusted US ETF OHLCV.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import pandas as pd
from quant_research import WalkForwardConfig, walk_forward, performance

SECTOR_ETFS = {
    "XLK": "Information Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}
BENCHMARK = {"SPY": "S&P 500 Benchmark"}
ALL_ASSETS = {**SECTOR_ETFS, **BENCHMARK}


def load_prices(ticker: str, source_dir: str | None = None,
                start: str = "2013-01-01") -> pd.DataFrame:
    if ticker not in ALL_ASSETS:
        raise ValueError(f"Unknown ETF {ticker}, must be one of {sorted(ALL_ASSETS)}")
    if source_dir:
        f = Path(source_dir)/f"{ticker}.csv"
        if not f.is_file():
            raise FileNotFoundError(f"No offline CSV for {ticker}: {f}")
        result = pd.read_csv(f)
    else:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install yfinance for downloads or use --data-dir") from exc
        result = yf.download(ticker, start=start, auto_adjust=True,
                             actions=False, progress=False, threads=False,
                             timeout=30)
        if result.empty:
            raise RuntimeError(f"No market history returned by yfinance for {ticker}")
        if isinstance(result.columns,pd.MultiIndex):
            result.columns=result.columns.get_level_values(0)
        result=result.reset_index()
        if "Datetime" in result.columns:
            result=result.rename(columns={"Datetime":"Date"})
    required={"Date","Open","High","Low","Close","Volume"}
    if not required.issubset(result.columns):
        raise ValueError(f"{ticker}: missing adjusted OHLCV fields: {required-set(result.columns)}")
    return result[list(("Date","Open","High","Low","Close","Volume"))].copy()


def evaluate_sector(ticker: str, config: WalkForwardConfig=WalkForwardConfig(),
                    source_dir: str | None=None, include_xgboost: bool=False,
                    start: str="2013-01-01") -> tuple[pd.DataFrame,dict,list]:
    raw=load_prices(ticker, source_dir=source_dir, start=start)
    predictions, choices=walk_forward(raw,config,include_xgboost=include_xgboost)
    details,stats=performance(predictions,config,initial_cash=50000)
    stats.update({"ticker":ticker,"sector":ALL_ASSETS[ticker],
                  "price_first":str(pd.to_datetime(raw["Date"].min()).date()),
                  "price_last":str(pd.to_datetime(raw["Date"].max()).date()),
                  "price_rows":len(raw),
                  "source":"offline_csv" if source_dir else "yfinance_adjusted_ohlcv",
                  "model_selections":[{"start":f["test_first_date"],
                                       "name":f["chosen_model"]["model"],
                                       "validation_rmse":f["chosen_model"]["validation_rmse"]}
                                      for f in choices]})
    return details,stats,choices


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--tickers",nargs="+",default=list(ALL_ASSETS))
    parser.add_argument("--data-dir",default=None,help="Offline raw adjusted OHLCV ETF CSVs")
    parser.add_argument("--start",default="2013-01-01")
    parser.add_argument("--oos",type=int,default=252,help="Historical out-of-sample sessions")
    parser.add_argument("--retrain",type=int,default=63)
    parser.add_argument("--min-train",type=int,default=504)
    parser.add_argument("--validation",type=int,default=126)
    parser.add_argument("--xgboost",action="store_true",help="Add XGBRegressor configurations")
    parser.add_argument("--out",default="output/sector_study")
    args=parser.parse_args()
    unknown=[t for t in args.tickers if t not in ALL_ASSETS]
    if unknown:
        parser.error(f"Unknown ETF tickers {unknown}")
    config=WalkForwardConfig(min_train=args.min_train,validation_size=args.validation,
                             test_size=args.oos,retrain_every=args.retrain)
    config.validate()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    metrics=[]
    failures=[]
    for ticker in args.tickers:
        print(f"Evaluating {ticker} ({ALL_ASSETS[ticker]})",flush=True)
        try:
            details,stats,selections=evaluate_sector(
                ticker,config,source_dir=args.data_dir,include_xgboost=args.xgboost,
                start=args.start)
            details.to_csv(out/f"{ticker}_oos.csv",index=False)
            (out/f"{ticker}_tuning.json").write_text(
                json.dumps(selections,indent=2,default=str),encoding="utf-8")
            metrics.append(stats)
            print(f"{ticker} research net={stats['net_return_pct']:.2f}%, "
                  f"allocated buy-hold={stats['allocated_buy_hold_return_pct']:.2f}%, "
                  f"oos={stats['n_oos_sessions']}, screen={stats['passes_research_screen']}",flush=True)
        except (ValueError,RuntimeError,OSError,KeyError) as exc:
            failures.append({"ticker":ticker,"error":f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {ticker}: {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
    report={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "research_only":True,
        "capital_usd":50000,
        "prediction_target":"next_open_to_following_open_adjusted_return",
        "model_selection":"chronological_train_validation_inner_fold_plus_outer_walk_forward",
        "backtest_model":"fractional_long_or_cash_at_next_open_with_assumed_turnover_fees",
        "notes":["Historical modeled orders, not broker fills or verified profits",
                 "Spread/market impact approximated by one-way bps",
                 "ETFs proxy sectors; evaluate constituent stocks separately",
                 "Model choice on validation; untouched outer fold for scoring",
                 "Data vendor changes/revisions and survivorship may affect results",
                 "No execution approvals are inferred from statistical screens"],
        "config":config.__dict__,
        "results":metrics,"failures":failures,
    }
    (out/"summary.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    if metrics:
        pd.DataFrame(metrics).drop(columns=["model_selections"]).to_csv(
            out/"sector_comparison.csv",index=False)
    print(f"Summary saved to {out/'summary.json'}; assets={len(metrics)}, errors={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
