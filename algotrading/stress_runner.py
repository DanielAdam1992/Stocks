"""Post-selection stress testing: research only, NOT a new independent holdout.

Selection of symbols based on earlier results introduces selection bias.
This intentionally tests longer periods and higher assumed trading costs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from quant_research import WalkForwardConfig, walk_forward, performance
from sector_runner import ALL_ASSETS, load_prices


def stress_symbol(ticker: str, horizons=(252,504,756),
                  cost_bps=(10.,20.,40.), start="2013-01-01",
                  data_dir=None, include_xgboost=True) -> list[dict]:
    raw=load_prices(ticker,source_dir=data_dir,start=start)
    results=[]
    for horizon in horizons:
        c=WalkForwardConfig(test_size=horizon,retrain_every=63)
        pred,_=walk_forward(raw,c,include_xgboost=include_xgboost)
        for bps in cost_bps:
            # Trading costs are one-way fee plus slippage; split equally here.
            stressed=replace(c,fee_bps_one_way=bps/2,
                             slippage_bps_one_way=bps/2)
            _,stats=performance(pred,stressed)
            results.append({
                "ticker":ticker,"industry":ALL_ASSETS[ticker],
                "horizon_requested":horizon,
                "total_one_way_bps":bps,
                **{k:v for k,v in stats.items()
                   if k in ("n_oos_sessions","start","end","net_return_pct",
                            "allocated_buy_hold_return_pct","full_buy_hold_return_pct",
                            "max_drawdown_pct","sharpe","n_entries","n_rebalances",
                            "mae_return","zero_forecast_mae","passes_research_screen")}
            })
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tickers",nargs="+",default=["V","PG","LIN","XLK"])
    p.add_argument("--data-dir",default=None)
    p.add_argument("--out",default="output/stress_study")
    p.add_argument("--no-xgboost",action="store_true")
    args=p.parse_args()
    unknown=[x for x in args.tickers if x not in ALL_ASSETS]
    if unknown: p.error(f"Unsupported symbols: {unknown}")
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    results=[]
    errors=[]
    for symbol in args.tickers:
        try:
            result=stress_symbol(symbol,data_dir=args.data_dir,
                                 include_xgboost=not args.no_xgboost)
            results.extend(result)
            for x in result:
                print(f"{x['ticker']} horizon={x['n_oos_sessions']} bps={x['total_one_way_bps']:.0f} "
                      f"net={x['net_return_pct']:.2f} passive={x['allocated_buy_hold_return_pct']:.2f} "
                      f"screen={x['passes_research_screen']}",flush=True)
        except (ValueError,RuntimeError,OSError,KeyError) as exc:
            errors.append({"ticker":symbol,"error":str(exc)})
    (out/"stress_results.json").write_text(json.dumps({
        "asof_utc":datetime.now(timezone.utc).isoformat(),
        "post_selection":True,
        "not_independent_holdout":True,
        "hypothesis":"Longer OOS windows and higher one-way cost sensitivity",
        "research_only":True,"results":results,"errors":errors},indent=2),encoding="utf-8")
    if results:
        pd.DataFrame(results).to_csv(out/"stress_results.csv",index=False)
    if errors:
        print(errors)
        raise SystemExit(1)


if __name__=="__main__":
    main()
