"""As-of quant research agent: fresh prediction, NOT a trade authorization.

Retrains a time-aware forecast from adjusted OHLCV up to latest available close.
Never loads stale notebook weights or sends broker orders.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from firm import Research, committee_decision
from bull_bear import investment_debate
from quant_research import FEATURES, make_features, choose_model, model_factory
from sector_runner import ALL_ASSETS, load_prices


def asof_signal(raw: pd.DataFrame, ticker: str, asof: date | None=None,
                min_train: int=504, validation_size: int=126,
                prediction_threshold: float=0.0015,
                include_xgboost: bool=False) -> dict:
    from quant_research import candidate_configs
    price_features=make_features(raw,include_unlabeled=True)
    labeled=price_features.dropna(subset=["target_return"]).reset_index(drop=True)
    if len(labeled)<min_train+validation_size+2:
        raise ValueError("Insufficient resolved return labels")
    decision=price_features.iloc[-1]
    asof_date=decision["Date"].date()
    if asof and asof_date>asof:
        raise ValueError("Market data contains observations after requested as-of")
    # Model history ends two sessions earlier, because the latest labels require
    # a second following opening price before being observed.
    if labeled["Date"].max() >= decision["Date"]:
        raise AssertionError("Unknown future returns entered supervised dataset")
    best,rankings=choose_model(labeled,validation_size,1,
                               candidate_configs(include_xgboost))
    model=model_factory(best["model"],best["params"])
    model.fit(labeled[list(FEATURES)],labeled["target_return"])
    estimate=float(model.predict(price_features.iloc[[-1]][list(FEATURES)])[0])
    payload={
        "ticker":ticker.upper(),"model":best["model"],
        "parameters":best["params"],
        "model_trained_from_scratch":True,
        "forecast_asof_market_close":asof_date.isoformat(),
        "target":"adjusted next open to following open return",
        "train_last_feature_date":labeled["Date"].iloc[-1].date().isoformat(),
        "training_rows":len(labeled),
        "validation_rmse":best["validation_rmse"],
        "candidate_validation_scores":rankings,
        "predicted_return":estimate,
        "proposed_direction":"RESEARCH_BUY" if estimate>prediction_threshold else "RESEARCH_HOLD",
        "approval_eligible":False,
        "reason":"No independent long-horizon validation, complete fundamentals/news, or portfolio risk approval",
        "execution_allowed":False,
    }
    return payload


def run_agent(ticker: str, asof: date | None=None, data_dir: str | None=None,
              include_xgboost: bool=False) -> dict:
    if ticker not in ALL_ASSETS:
        raise ValueError("Unsupported ticker")
    raw=load_prices(ticker,source_dir=data_dir)
    quant=asof_signal(raw,ticker,asof=asof,include_xgboost=include_xgboost)
    # Hand the forecast to CIO for record keeping; do not fabricate completed
    # financial or news research. Independent risk officer remains in veto.
    record=Research(ticker=ticker,asof=quant["forecast_asof_market_close"],
                    predicted_return=quant["predicted_return"],
                    financials_reviewed=False,news_reviewed=False,
                    evidence=(),red_flags=("quant_only_no_fundamental_or_news_review",))
    d=committee_decision(record)
    return {
        "research_mode":"PAPER_ONLY",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "quant_research":quant,
        "bull_bear_analysis":investment_debate(ticker,quant,financial=None),
        "committee":{"action":d.action,"risk_approved":d.risk_approved,
                     "rationale":d.rationale},
        "broker_orders":[],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("ticker",choices=sorted(ALL_ASSETS))
    p.add_argument("--asof",default=None,help="Optional latest allowed market date YYYY-MM-DD")
    p.add_argument("--data-dir",default=None)
    p.add_argument("--xgboost",action="store_true")
    p.add_argument("--out",default="output/quant_agent")
    args=p.parse_args()
    date_cutoff=date.fromisoformat(args.asof) if args.asof else None
    result=run_agent(args.ticker,asof=date_cutoff,data_dir=args.data_dir,
                     include_xgboost=args.xgboost)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    path=out/f"{args.ticker}_signal.json"
    path.write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(json.dumps({"ticker":args.ticker,
                      "asof":result["quant_research"]["forecast_asof_market_close"],
                      "signal":result["quant_research"]["proposed_direction"],
                      "model":result["quant_research"]["model"],
                      "committee":result["committee"]["action"],
                      "risk_approved":result["committee"]["risk_approved"],
                      "orders":len(result["broker_orders"]),
                      "record":str(path)},indent=2))


if __name__=="__main__":
    main()
