"""Leakage-aware, walk-forward *research* forecasts for US equities/ETFs.

Features observed by the close of session t; predicted outcome is the
adjusted-open(t+1) -> adjusted-open(t+2) return. Signal executes at next open.
No look-ahead features, no global scaling, no random shuffle and no broker.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = (
    "return_1d", "momentum_5d", "momentum_10d", "momentum_20d",
    "momentum_60d", "volatility_5d", "volatility_20d",
    "intraday_return", "overnight_gap", "high_low_range", "volume_ratio_20d",
)
SEED = 42
DEFAULT_CONFIGS = (
    ("zero", {}),
    ("ridge", {"alpha": 10.0}),
    ("ridge", {"alpha": 100.0}),
    ("histgb", {"max_leaf_nodes": 7, "learning_rate": 0.03, "max_iter": 90,
                "min_samples_leaf": 35, "l2_regularization": 1.0}),
    ("histgb", {"max_leaf_nodes": 15, "learning_rate": 0.04, "max_iter": 110,
                "min_samples_leaf": 45, "l2_regularization": 5.0}),
)


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train: int = 504
    validation_size: int = 126
    test_size: int = 252
    retrain_every: int = 63
    # With open(t+1)->open(t+2) label, last train label at t-2
    # is already observable at the close of t, but t-1 is not.
    purge_rows: int = 1
    signal_threshold: float = 0.0015
    max_weight: float = 0.10
    fee_bps_one_way: float = 5.0
    slippage_bps_one_way: float = 5.0

    def validate(self) -> None:
        if self.min_train < 50 or self.validation_size < 20 or self.test_size < 5:
            raise ValueError("Insufficient training/validation/test rows")
        if self.retrain_every < 1 or self.purge_rows < 1:
            raise ValueError("Invalid retrain or purge interval")
        if not (0 < self.max_weight <= 0.25):
            raise ValueError("max_weight must lie in (0, 0.25]")
        if min(self.signal_threshold, self.fee_bps_one_way,
               self.slippage_bps_one_way) < 0:
            raise ValueError("Negative trading thresholds/costs not permitted")


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Accept Date,Open,High,Low,Close,Volume; assume prices adjusted consistently."""
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing OHLCV columns: {sorted(required-set(raw.columns))}")
    df = raw[list(required)].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise", utc=True).dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)
    if df["Date"].duplicated().any():
        raise ValueError("Duplicate price dates")
    for c in required - {"Date"}:
        df[c] = pd.to_numeric(df[c], errors="raise")
    if not np.isfinite(df[list(required - {"Date"})].to_numpy(float)).all():
        raise ValueError("Missing/invalid OHLCV; never auto-backfill")
    if (df[["Open","High","Low","Close"]] <= 0).any().any() or (df["Volume"] < 0).any():
        raise ValueError("Nonpositive prices or negative volume")
    # Most vendors round separately adjusted OHLC prices to different floating precision.
    # A small tolerance is allowed, but significant violations still fail closed.
    upper=df[["Open","Close","Low"]].max(axis=1)
    lower=df[["Open","Close","High"]].min(axis=1)
    upper_excess=(upper-df["High"]).clip(lower=0)
    lower_excess=(df["Low"]-lower).clip(lower=0)
    tolerance=1e-5*df["Close"]
    violations=(upper_excess>tolerance)|(lower_excess>tolerance)
    if violations.any():
        max_error=float((pd.concat([upper_excess,lower_excess],axis=1).max(axis=1) /
                         df["Close"]).max())
        raise ValueError(f"OHLC price bounds invalid: rows={int(violations.sum())}, "
                         f"max_relative_error={max_error:.8%}")

    close = df["Close"]
    df["return_1d"] = close.pct_change()
    for d in (5,10,20,60):
        df[f"momentum_{d}d"] = close.pct_change(d)
    for d in (5,20):
        df[f"volatility_{d}d"] = close.pct_change().rolling(d).std()
    df["intraday_return"] = close/df["Open"]-1
    df["overnight_gap"] = df["Open"]/close.shift(1)-1
    df["high_low_range"] = (df["High"]-df["Low"])/close
    volume_mean = df["Volume"].rolling(20).mean()
    df["volume_ratio_20d"] = df["Volume"]/volume_mean.replace(0,np.nan)-1

    # Signal from close t trades at open t+1, exits at open t+2.
    df["entry_open"] = df["Open"].shift(-1)
    df["exit_open"] = df["Open"].shift(-2)
    df["target_return"] = df["exit_open"]/df["entry_open"]-1
    df = df.replace([np.inf,-np.inf],np.nan)
    # Leading warm-up rows and trailing unknown targets are dropped;
    # no backfill or forward-fill from future prices.
    df = df.dropna(subset=[*FEATURES,"target_return"]).reset_index(drop=True)
    return df


def model_factory(name: str, params: dict):
    if name == "zero":
        return DummyRegressor(strategy="constant",constant=0.0)
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    if name == "histgb":
        return HistGradientBoostingRegressor(random_state=SEED,early_stopping=False,**params)
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("Install xgboost to enable legacy XGB benchmarks") from exc
        return XGBRegressor(random_state=SEED,n_jobs=1,objective="reg:squarederror",**params)
    raise ValueError(f"Unknown model {name}")


def candidate_configs(include_xgboost: bool=False):
    configs = list(DEFAULT_CONFIGS)
    if include_xgboost:
        configs.extend((
            ("xgboost", {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.03,
                         "subsample": .8, "colsample_bytree": .8, "reg_lambda": 10}),
            ("xgboost", {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.03,
                         "subsample": .8, "colsample_bytree": .8, "reg_lambda": 20}),
        ))
    return configs


def choose_model(hist: pd.DataFrame, validation_size: int,
                 purge_rows: int = 1, configs=None):
    """Tune ONLY inside historical training data; never use the future test block."""
    configs = list(configs if configs is not None else DEFAULT_CONFIGS)
    if not configs:
        raise ValueError("No candidate configs")
    n = len(hist)
    valid_start = n-validation_size
    fit_end = valid_start-purge_rows
    if fit_end < 50:
        raise ValueError("Insufficient time-aware training history")
    X_train = hist.loc[:fit_end-1, list(FEATURES)]
    y_train = hist.loc[:fit_end-1,"target_return"]
    X_valid = hist.loc[valid_start:, list(FEATURES)]
    y_valid = hist.loc[valid_start:,"target_return"]
    rankings = []
    for name,params in configs:
        model = model_factory(name,params)
        model.fit(X_train,y_train)
        pred = model.predict(X_valid)
        rmse = float(np.sqrt(np.mean((y_valid.to_numpy()-pred)**2)))
        mae = float(np.mean(abs(y_valid.to_numpy()-pred)))
        rankings.append({"model":name, "params":params, "validation_rmse":rmse,
                         "validation_mae":mae})
    # Rank by RMSE; complexity tie breaks favor a simpler model, and stable ordering.
    best = min(rankings,key=lambda r:r["validation_rmse"])
    return best,rankings


def walk_forward(prices: pd.DataFrame, config: WalkForwardConfig=WalkForwardConfig(),
                 include_xgboost: bool=False):
    config.validate()
    df = make_features(prices)
    n = len(df)
    earliest = config.min_train+config.validation_size+2*config.purge_rows
    start = max(earliest,n-config.test_size)
    if start >= n:
        raise ValueError(f"Insufficient history: {n} feature-ready rows; require > {earliest}")
    results = []
    selections = []
    configs = candidate_configs(include_xgboost)
    for pos in range(start,n,config.retrain_every):
        # Training labels must end strictly before the forecast block.
        history = df.iloc[:pos-config.purge_rows].reset_index(drop=True)
        if len(history) < config.min_train+config.validation_size:
            raise ValueError("Too little train history after purge")
        best,rankings = choose_model(history,config.validation_size,config.purge_rows,configs)
        model = model_factory(best["model"],best["params"])
        model.fit(history[list(FEATURES)],history["target_return"])
        test = df.iloc[pos:min(pos+config.retrain_every,n)].copy()
        test["predicted_return"] = model.predict(test[list(FEATURES)])
        test["model"] = best["model"]
        test["train_end_date"] = history["Date"].iloc[-1]
        test["validation_rmse"] = best["validation_rmse"]
        results.append(test)
        selections.append({"test_first_date":test["Date"].iloc[0].date().isoformat(),
                           "train_end_date":history["Date"].iloc[-1].date().isoformat(),
                           "chosen_model":best,"candidates":rankings})
    preds = pd.concat(results,ignore_index=True)
    return preds,selections


def performance(preds: pd.DataFrame, config: WalkForwardConfig=WalkForwardConfig(),
                initial_cash: float=50000.0) -> tuple[pd.DataFrame,dict]:
    """Opening auction is an approximation; no actual order book or spread sampling."""
    if initial_cash<=0:
        raise ValueError("Invalid capital")
    df = preds.copy().reset_index(drop=True)
    required={"Date","entry_open","exit_open","predicted_return","target_return"}
    if not required.issubset(df.columns):
        raise ValueError("Missing prediction or price columns")
    if df.empty or df[list(required-{"Date"})].isna().any().any():
        raise ValueError("Empty or incomplete predictions")
    if (df[["entry_open","exit_open"]]<=0).any().any():
        raise ValueError("Invalid executable prices")
    df["weight"]=np.where(df["predicted_return"]>config.signal_threshold,config.max_weight,0.0)
    # Rebalance for the next opening session. Trading cost approximates fees + impact.
    df["turnover"]=df["weight"].diff().abs()
    df.loc[0,"turnover"]=abs(df.loc[0,"weight"])
    total_bps=config.fee_bps_one_way+config.slippage_bps_one_way
    df["gross_return"]=df["weight"]*df["target_return"]
    df["net_return"]=df["gross_return"]-df["turnover"]*total_bps/10000
    df["equity"]=initial_cash*(1+df["net_return"]).cumprod()
    df["baseline_allocated_return"]=config.max_weight*df["target_return"]
    df["baseline_allocated_equity"]=initial_cash*(1+df["baseline_allocated_return"]).cumprod()
    df["baseline_full_equity"]=initial_cash*(1+df["target_return"]).cumprod()
    eq=df["equity"]
    daily=df["net_return"]
    annual_sharpe=(float(np.sqrt(252)*daily.mean()/daily.std(ddof=1))
                   if len(daily)>1 and daily.std(ddof=1)>0 else None)
    hit=float((np.sign(df["predicted_return"])==np.sign(df["target_return"])).mean())
    drawdown=float((eq/eq.cummax()-1).min())
    stats={
        "n_oos_sessions":int(len(df)),
        "start":str(df["Date"].iloc[0].date()),
        "end":str(df["Date"].iloc[-1].date()),
        "initial_cash_usd":initial_cash,
        "ending_equity_usd":float(eq.iloc[-1]),
        "net_return_pct":float(100*(eq.iloc[-1]/initial_cash-1)),
        "allocated_buy_hold_return_pct":float(100*(df["baseline_allocated_equity"].iloc[-1]/initial_cash-1)),
        "full_buy_hold_return_pct":float(100*(df["baseline_full_equity"].iloc[-1]/initial_cash-1)),
        "max_drawdown_pct":100*drawdown,
        "sharpe":annual_sharpe,
        "direction_hit_rate":hit,
        "mae_return":float((df["predicted_return"]-df["target_return"]).abs().mean()),
        "zero_forecast_mae":float(df["target_return"].abs().mean()),
        "n_entries":int(((df["weight"]>0)&(df["weight"].shift(1).fillna(0)==0)).sum()),
        "n_rebalances":int((df["turnover"]>0).sum()),
        "total_turnover":float(df["turnover"].sum()),
        "assumed_one_way_bps":total_bps,
    }
    stats["passes_research_screen"]=bool(
        stats["n_oos_sessions"]>=200 and stats["n_entries"]>=10
        and stats["mae_return"]<stats["zero_forecast_mae"]
        and stats["net_return_pct"]>stats["allocated_buy_hold_return_pct"]
        and stats["sharpe"] is not None and stats["sharpe"]>0
    )
    # Screen is only a shortlist criterion and NEVER execution approval.
    return df,stats
