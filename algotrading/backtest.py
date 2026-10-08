"""Research-only, long/cash backtest. Signals are decided after close t and earn t+1 close-to-close return.
Input CSV: Date, Close, predicted_return. predicted_return must be genuinely out-of-sample
and generated using information available no later than that row's close.
No broker connectivity or live trading.
"""
import argparse
import pandas as pd
import numpy as np

def backtest(frame, initial_cash=50000.0, threshold=0.005, fee_bps=5.0, max_weight=0.25):
    if not (0 < max_weight <= 1): raise ValueError("max_weight must be in (0,1]")
    if initial_cash <= 0 or fee_bps < 0 or threshold < 0: raise ValueError("invalid parameters")
    required = {"Date", "Close", "predicted_return"}
    if not required.issubset(frame.columns): raise ValueError(f"Missing columns: {required-set(frame.columns)}")
    df = frame.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values("Date")
    if df["Date"].duplicated().any(): raise ValueError("duplicate dates")
    for col in ("Close", "predicted_return"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df[["Close", "predicted_return"]].isna().any().any() or (df["Close"] <= 0).any(): raise ValueError("invalid prices or predictions")
    if len(df) < 3: raise ValueError("at least three rows required")
    # At close t choose next session's exposure. First session starts in cash.
    df["target_weight"] = np.where(df["predicted_return"] > threshold, max_weight, 0.0)
    df["weight"] = df["target_weight"].shift(1).fillna(0.0)
    df["asset_return"] = df["Close"].pct_change().fillna(0.0)
    # Approximation: rebalance cost from weight changes, not drift-adjusted holdings.
    df["turnover"] = df["weight"].diff().abs().fillna(0.0)
    df["strategy_return"] = df["weight"] * df["asset_return"] - df["turnover"] * fee_bps / 10000.0
    df["equity"] = initial_cash * (1 + df["strategy_return"]).cumprod()
    df["buy_hold_equity"] = initial_cash * (df["Close"] / df["Close"].iloc[0])
    peak = df["equity"].cummax()
    max_dd = float((df["equity"] / peak - 1).min())
    r = df["strategy_return"]
    sharpe = float(np.sqrt(252)*r.mean()/r.std()) if r.std() > 0 else None
    stats = {"starting_cash": initial_cash, "ending_equity": float(df["equity"].iloc[-1]),
             "net_return_pct": float((df["equity"].iloc[-1]/initial_cash-1)*100),
             "buy_hold_return_pct": float((df["buy_hold_equity"].iloc[-1]/initial_cash-1)*100),
             "max_drawdown_pct": max_dd*100, "sharpe_annualized": sharpe,
             "trading_days": len(df)-1, "rebalances": int((df["turnover"]>0).sum())}
    return df, stats

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("predictions_csv", help="Date,Close,predicted_return (walk-forward OOS predictions)")
    p.add_argument("--output", default="backtest_results.csv")
    p.add_argument("--threshold", type=float, default=0.005)
    p.add_argument("--fee-bps", type=float, default=5)
    p.add_argument("--max-weight", type=float, default=0.25)
    a = p.parse_args()
    result, stats = backtest(pd.read_csv(a.predictions_csv), threshold=a.threshold, fee_bps=a.fee_bps, max_weight=a.max_weight)
    result.to_csv(a.output, index=False)
    print(pd.Series(stats).to_string())
