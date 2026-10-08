"""Institutional-style research portfolio laboratory (NO broker access).

Research signals are computed after close t and rebalanced at open t+1.
The holding P&L is open(t+1) -> open(t+2), so no signal earns a
return before it can be acted on. All dates are US market sessions.

This is a fractional-share, adjusted-price simulation. Tax, dividend
payment timing, bid/ask order book, and actual broker fills are not modeled.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import sqrt
from typing import Mapping

import numpy as np
import pandas as pd

SECTOR_ETFS = (
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
    "XLP", "XLU", "XLB", "XLRE", "XLC"
)
MARKET = "SPY"
MODES = ("sector_rotation", "cross_sectional_momentum", "sector_mean_reversion")


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 50_000.0
    rebalance_every: int = 5
    top_n: int = 5
    max_position_weight: float = 0.10
    max_sector_weight: float = 0.20
    max_gross_weight: float = 0.50
    max_drawdown: float = 0.12
    one_way_fee_bps: float = 5.0
    one_way_slippage_bps: float = 5.0
    trend_window: int = 200
    volatility_window: int = 20
    volatility_ceiling: float = 0.40
    test_sessions: int = 504

    def validate(self) -> None:
        if self.initial_capital <= 0 or self.rebalance_every < 1 or self.top_n < 1:
            raise ValueError("Invalid capital, top_n or rebalance interval")
        if not (0 < self.max_position_weight <= self.max_sector_weight <=
                self.max_gross_weight <= 1):
            raise ValueError("Invalid position/sector/gross concentration limits")
        if not (0 < self.max_drawdown < 1):
            raise ValueError("Invalid max drawdown threshold")
        if self.trend_window < 25 or self.volatility_window < 5 or self.test_sessions < 10:
            raise ValueError("Insufficient market history windows")
        if min(self.one_way_fee_bps, self.one_way_slippage_bps) < 0:
            raise ValueError("Transaction costs must be nonnegative")


@dataclass
class MarketPanel:
    dates: pd.DatetimeIndex
    open: pd.DataFrame
    close: pd.DataFrame
    snapshots_sha256: str


def prepare_market_panel(history: Mapping[str,pd.DataFrame]) -> MarketPanel:
    """Require complete synchronous US-session prices; never forward/back-fill.

    A fixed, retrospective symbol universe is subject to survivorship bias.
    Market-data snapshots are hashed so runs can be audited.
    """
    if MARKET not in history or len(history) < 2:
        raise ValueError("Require SPY market benchmark and at least one tradable asset")
    normalized = {}
    h = hashlib.sha256()
    for symbol in sorted(history):
        if not symbol or not isinstance(symbol,str):
            raise ValueError("Invalid symbol")
        raw=history[symbol]
        required={"Date","Open","Close"}
        if not required.issubset(raw.columns):
            raise ValueError(f"{symbol}: missing {required-set(raw.columns)}")
        frame=raw[["Date","Open","Close"]].copy()
        frame["Date"]=pd.to_datetime(frame["Date"],errors="raise",utc=True).dt.tz_localize(None)
        frame=frame.sort_values("Date")
        if frame["Date"].duplicated().any():
            raise ValueError(f"{symbol}: duplicate session dates")
        for field in ("Open","Close"):
            frame[field]=pd.to_numeric(frame[field],errors="raise")
        values=frame[["Open","Close"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values<=0):
            raise ValueError(f"{symbol}: invalid adjusted prices")
        frame=frame.set_index("Date")
        if len(frame)<50:
            raise ValueError(f"{symbol}: fewer than 50 sessions")
        h.update(symbol.encode())
        h.update(frame.to_csv(float_format="%.10g").encode())
        normalized[symbol]=frame
    # Inner join prevents stale quotes and silent imputation. Track loss of dates.
    common=None
    for f in normalized.values():
        common=f.index if common is None else common.intersection(f.index)
    common=common.sort_values()
    if len(common)<50:
        raise ValueError("Too few common market sessions")
    if (common.max()-common.min()).days>len(common)*3:
        raise ValueError("Suspiciously sparse aligned market history")
    opens=pd.DataFrame({s:f.loc[common,"Open"].to_numpy(dtype=float)
                        for s,f in normalized.items()},index=common)
    closes=pd.DataFrame({s:f.loc[common,"Close"].to_numpy(dtype=float)
                         for s,f in normalized.items()},index=common)
    return MarketPanel(dates=common,open=opens,close=closes,
                       snapshots_sha256=h.hexdigest())


def regime_frame(spy_close: pd.Series, cfg: PortfolioConfig) -> pd.DataFrame:
    """Regime known only at each close: no full-sample percentiles."""
    cfg.validate()
    trend=spy_close.rolling(cfg.trend_window,min_periods=cfg.trend_window).mean()
    vol=spy_close.pct_change().rolling(
        cfg.volatility_window,min_periods=cfg.volatility_window).std()*sqrt(252)
    category=np.where(trend.isna()|vol.isna(),"unknown",
               np.where(vol>cfg.volatility_ceiling,"high_volatility",
               np.where(spy_close>trend,"risk_on","risk_off")))
    return pd.DataFrame({"market_close":spy_close,"market_trend":trend,
                         "realized_vol_annualized":vol,
                         "regime":category},index=spy_close.index)


def _signals(panel: MarketPanel, cfg: PortfolioConfig) -> dict[str,pd.DataFrame]:
    close=panel.close.drop(columns=[MARKET],errors="ignore")
    returns=close.pct_change()
    return {
        "mom_21":close.pct_change(21),
        "mom_63":close.pct_change(63),
        "mom_5":close.pct_change(5),
        "vol_20":returns.rolling(20,min_periods=20).std()*sqrt(252),
        "ma_100":close.rolling(100,min_periods=100).mean(),
        "ma_10":close.rolling(10,min_periods=10).mean(),
    }


def validate_allocation(weights: Mapping[str,float],
                        cfg: PortfolioConfig, sector_map: Mapping[str,str],
                        available: set[str]) -> None:
    """Independent risk officer veto: reject invalid signal weights."""
    sector_total={}
    gross=0.
    for ticker,w in weights.items():
        if ticker not in available or ticker==MARKET:
            raise ValueError("Unknown or forbidden portfolio symbol")
        if not np.isfinite(w) or w<0 or w>cfg.max_position_weight+1e-9:
            raise ValueError("Short/leverage/concentration not permitted")
        gross+=w
        s=sector_map.get(ticker)
        if not s:
            raise ValueError(f"Missing sector classification for {ticker}")
        sector_total[s]=sector_total.get(s,0.)+w
    if gross>cfg.max_gross_weight+1e-9 or gross>1+1e-9:
        raise ValueError("Gross exposure limit violated")
    if any(w>cfg.max_sector_weight+1e-9 for w in sector_total.values()):
        raise ValueError("Sector concentration limit violated")


def propose_allocations(panel: MarketPanel, strategy: str, cfg: PortfolioConfig,
                        sector_map: Mapping[str,str]) -> tuple[dict, pd.DataFrame]:
    """Create independent, fully deterministic strategy proposals.

    Every proposal at close t is an allocation *target*, not an order.
    Risk-off regimes go to cash; all assets are liquid US stock/ETF proxies.
    """
    cfg.validate()
    if strategy not in MODES:
        raise ValueError(f"Unsupported strategy: {strategy}")
    tickers=[s for s in panel.close.columns if s!=MARKET]
    if not tickers:
        raise ValueError("No eligible assets")
    if set(tickers)-set(sector_map):
        raise ValueError("Not every symbol has a sector classification")
    if strategy in ("sector_rotation","sector_mean_reversion") and \
       any(t not in SECTOR_ETFS for t in tickers):
        raise ValueError("Sector strategies must use the 11 sector ETFs")
    warmup=max(cfg.trend_window,100,63)+2
    if len(panel.dates) <= warmup+2:
        raise ValueError("Insufficient history for model-free strategies")
    metrics=_signals(panel,cfg)
    states=regime_frame(panel.close[MARKET],cfg)
    proposed={}
    logs=[]
    previous_regime=None
    for i in range(warmup,len(panel.dates)-2):
        dt=panel.dates[i]
        state=str(states.loc[dt,"regime"])
        scheduled=(i-warmup)%cfg.rebalance_every==0
        # A detected transition to risk-off triggers immediate exit at next open.
        change=(previous_regime is not None and state!=previous_regime)
        previous_regime=state
        if not (scheduled or change):
            continue
        selected={}
        reasons=[]
        scores=[]
        if state!="risk_on":
            reasons.append(f"Regime veto: {state}")
        else:
            for ticker in tickers:
                m21=float(metrics["mom_21"].loc[dt,ticker])
                m63=float(metrics["mom_63"].loc[dt,ticker])
                v=float(metrics["vol_20"].loc[dt,ticker])
                latest=float(panel.close.loc[dt,ticker])
                ma100=float(metrics["ma_100"].loc[dt,ticker])
                ma10=float(metrics["ma_10"].loc[dt,ticker])
                if not all(np.isfinite([m21,m63,v,latest,ma100,ma10])) or v<=0:
                    continue
                if strategy=="sector_rotation":
                    if m63<=0 or latest<ma100:
                        continue
                    score=(0.6*m63+0.4*m21)/max(v,0.08)
                elif strategy=="cross_sectional_momentum":
                    if m63<=0 or latest<ma100:
                        continue
                    score=(0.75*m63+0.25*m21)/max(v,0.08)
                else:
                    # Select pullbacks within established longer-term uptrends.
                    deviation=latest/ma10-1
                    if m63<=0 or deviation>=-0.015:
                        continue
                    score=-deviation/max(v/sqrt(252),0.005)
                scores.append((ticker,float(score)))
            scores.sort(key=lambda pair:(-pair[1],pair[0]))
            sector_usage={}
            for ticker,score in scores:
                if len(selected)>=cfg.top_n:
                    break
                if len(selected)*cfg.max_position_weight>=cfg.max_gross_weight-1e-9:
                    break
                sec=sector_map[ticker]
                if sector_usage.get(sec,0.)+cfg.max_position_weight>cfg.max_sector_weight+1e-9:
                    continue
                selected[ticker]=cfg.max_position_weight
                sector_usage[sec]=sector_usage.get(sec,0.)+cfg.max_position_weight
            if not selected:
                reasons.append("No symbols passed observable setup filters")
        validate_allocation(selected,cfg,sector_map,set(tickers))
        proposed[dt]=selected
        logs.append({"signal_close":dt.isoformat(),
                     "next_execution_open":panel.dates[i+1].isoformat(),
                     "regime":state,"strategy":strategy,
                     "selected":",".join(selected),
                     "gross_weight":sum(selected.values()),
                     "reason":"; ".join(reasons),
                     "eligible_scores":json.dumps(scores[:cfg.top_n*2])})
    return proposed,pd.DataFrame(logs)


def run_portfolio(panel: MarketPanel, allocations: Mapping, cfg: PortfolioConfig,
                  sector_map: Mapping[str,str]) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    """Simulate cash and simultaneous holdings with drift and one-way turnover.

    A decision dated D[t] is filled at D[t+1] adjusted open and P&L is
    measured through D[t+2] open. An untriggered date does NOT rebalance
    back to original weights. Fractional shares are assumed.
    """
    cfg.validate()
    all_assets=set(panel.open.columns)-{MARKET}
    for key,targets in allocations.items():
        if key not in panel.dates:
            raise ValueError("Target's close timestamp absent from market data")
        validate_allocation(targets,cfg,sector_map,all_assets)
    if not allocations:
        raise ValueError("No strategy rebalance decisions")
    dates=panel.dates
    first=dates.get_loc(min(allocations))
    if first+2>=len(dates):
        raise ValueError("No realizable next-open test interval")
    # Only score the last cfg.test_sessions decisions, with adequate warmup.
    start=max(first,len(dates)-cfg.test_sessions-2)
    eq=cfg.initial_capital
    high_water=eq
    held={s:0.0 for s in all_assets}
    halted=False
    trade_rows=[]
    bars=[]
    spy_equity=cfg.initial_capital
    allocated_spy_equity=cfg.initial_capital
    buy_and_hold_weight=cfg.max_gross_weight
    spy_entry_cost_paid=False
    cost_rate=(cfg.one_way_fee_bps+cfg.one_way_slippage_bps)/10_000.0
    for t in range(start,len(dates)-2):
        signal_date=dates[t]
        exec_date=dates[t+1]
        next_date=dates[t+2]
        before=eq
        peak_dd=eq/high_water-1
        if peak_dd<=-cfg.max_drawdown:
            halted=True
        targets=allocations.get(signal_date)
        if halted:
            targets={}
        turnover=0.0
        if targets is not None:
            # Rebalance current *drifted* portfolio. Only buy/sell at next open.
            for ticker in sorted(all_assets):
                before_w=held[ticker]
                target_w=targets.get(ticker,0.)
                delta=target_w-before_w
                if abs(delta)>1e-9:
                    trade_rows.append({
                        "decision_close":signal_date.isoformat(),
                        "execution_open":exec_date.isoformat(),
                        "ticker":ticker,
                        "side":"BUY" if delta>0 else "SELL",
                        "target_weight":target_w,
                        "weight_delta":delta,
                        "approx_notional_usd":abs(delta)*before,
                        "approx_one_way_cost_usd":abs(delta)*before*cost_rate,
                        "paper_only":True,
                    })
                    turnover+=abs(delta)
                held[ticker]=target_w
        # Estimated execution expenses at starting open, prior to holding period.
        execution_cost=before*turnover*cost_rate
        returns=(panel.open.loc[next_date,list(all_assets)]/
                 panel.open.loc[exec_date,list(all_assets)]-1).to_dict()
        gross_return=sum(held[ticker]*float(returns[ticker]) for ticker in all_assets)
        eq=(before-execution_cost)*(1+gross_return)
        if not np.isfinite(eq) or eq<=0:
            raise ValueError("Simulation bankrupt or invalid; reject")
        # Convert to drifted pre-rebalance weights at the next opening.
        for ticker in held:
            held[ticker]=held[ticker]*(1+float(returns[ticker]))/(1+gross_return)
        if sum(held.values())>1+1e-8:
            raise ValueError("Unexpected leverage after price drift")
        high_water=max(high_water,eq)
        spy_r=float(panel.open.loc[next_date,MARKET]/panel.open.loc[exec_date,MARKET]-1)
        spy_equity*=1+spy_r
        if not spy_entry_cost_paid:
            allocated_spy_equity*=(1-buy_and_hold_weight*cost_rate)
            spy_entry_cost_paid=True
        allocated_spy_equity*=1+buy_and_hold_weight*spy_r
        bars.append({
            "signal_close":signal_date.isoformat(),
            "execution_open":exec_date.isoformat(),
            "mark_next_open":next_date.isoformat(),
            "equity_usd":eq,
            "daily_net_return":eq/before-1,
            "gross_return":gross_return,
            "transaction_cost_usd":execution_cost,
            "turnover":turnover,
            "gross_exposure_at_open":sum(targets.values()) if targets is not None and not halted
                                     else (sum(held.values()) if targets is None else 0.),
            "cash_usd_approx":eq*(1-sum(held.values())),
            "max_drawdown_so_far":eq/high_water-1,
            "risk_halted":halted,
            "spy_full_equity":spy_equity,
            "spy_risk_budget_equity":allocated_spy_equity,
        })
    curve=pd.DataFrame(bars)
    trades=pd.DataFrame(trade_rows,columns=[
        "decision_close","execution_open","ticker","side","target_weight",
        "weight_delta","approx_notional_usd","approx_one_way_cost_usd","paper_only"])
    rets=curve["daily_net_return"]
    sharpe=(float(sqrt(252)*rets.mean()/rets.std(ddof=1))
            if len(rets)>1 and rets.std(ddof=1)>1e-12 else None)
    max_dd=float((curve["equity_usd"]/curve["equity_usd"].cummax()-1).min())
    summary={
        "paper_research_only":True,
        "tested_from":curve.iloc[0]["signal_close"],
        "tested_to":curve.iloc[-1]["signal_close"],
        "n_sessions":len(curve),
        "initial_capital_usd":cfg.initial_capital,
        "final_equity_usd":float(eq),
        "net_return_pct":float((eq/cfg.initial_capital-1)*100),
        "spy_full_return_pct":float((spy_equity/cfg.initial_capital-1)*100),
        "spy_risk_budget_return_pct":float((allocated_spy_equity/cfg.initial_capital-1)*100),
        "max_drawdown_pct":100*max_dd,
        "annualized_sharpe":sharpe,
        "order_events":len(trades),
        "rebalance_days":int((curve["turnover"]>1e-9).sum()),
        "total_turnover":float(curve["turnover"].sum()),
        "estimated_transaction_cost_usd":float(curve["transaction_cost_usd"].sum()),
        "risk_stop_triggered":bool(halted),
        "market_price_snapshot_sha256":panel.snapshots_sha256,
        "configuration":asdict(cfg),
        "assumptions":[
            "Adjusted prices, no actual fills or brokerage account",
            "Fractional shares, prices at next open, fixed total cost in bps",
            "Cash earns zero interest, tax and fees beyond specified bps excluded",
            "Fixed retrospectively selected symbols; survivorship and regime bias remain",
            "Historical simulation does not establish future profitability"
        ],
    }
    return curve,trades,summary
