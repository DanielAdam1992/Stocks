"""Synthetic, deterministic tests. Synthetic gains are NOT investment results."""
from datetime import timedelta
import numpy as np
import pandas as pd
import pytest

from portfolio_lab import (PortfolioConfig, SECTOR_ETFS, prepare_market_panel,
                           propose_allocations, regime_frame, run_portfolio,
                           validate_allocation)
from bull_bear import investment_debate


def fake_market(n=330):
    dates=pd.bdate_range("2024-01-01",periods=n)
    step=np.arange(n,dtype=float)
    result={}
    for i,ticker in enumerate(("SPY","XLK","XLF","XLE","XLV","XLI",
                               "AAPL","MSFT","JPM","V","XOM")):
        close=100*np.exp((.00055+i*.00006)*step + .01*np.sin(step/11+i))
        op=np.roll(close,1)
        op[0]=close[0]
        result[ticker]=pd.DataFrame({
            "Date":dates,"Open":op,"Close":close,
            "High":np.maximum(op,close)*1.002,
            "Low":np.minimum(op,close)*.998,
            "Volume":np.full(n,1_000_000+i),
        })
    return result


def small_config(**kwargs):
    return PortfolioConfig(test_sessions=80,trend_window=50,
                           volatility_window=10,**kwargs)


def test_market_snapshot_and_no_price_imputation():
    history=fake_market()
    panel=prepare_market_panel(history)
    assert len(panel.dates)==330
    assert len(panel.snapshots_sha256)==64
    history["XLK"].loc[5,"Open"]=np.nan
    with pytest.raises(ValueError):
        prepare_market_panel(history)


def test_regime_uses_only_observed_close_prices():
    panel=prepare_market_panel(fake_market())
    cfg=small_config()
    original=regime_frame(panel.close["SPY"],cfg)
    changed=panel.close["SPY"].copy()
    changed.iloc[-1]*=0.25
    altered=regime_frame(changed,cfg)
    pd.testing.assert_frame_equal(original.iloc[:-1],altered.iloc[:-1])


def test_sector_rotation_rebalances_after_close_and_respects_limits():
    symbols=["SPY","XLK","XLF","XLE","XLV","XLI"]
    history=fake_market()
    panel=prepare_market_panel({s:history[s] for s in symbols})
    cfg=small_config()
    secmap={s:s for s in symbols if s!="SPY"}
    allocations,log=propose_allocations(panel,"sector_rotation",cfg,secmap)
    assert allocations
    assert not log.empty
    for ts,weights in allocations.items():
        validate_allocation(weights,cfg,secmap,set(secmap))
        matching=log[log["signal_close"]==ts.isoformat()]
        assert len(matching)==1
        assert pd.Timestamp(matching.iloc[0]["next_execution_open"]) > ts
    curve,trades,summary=run_portfolio(panel,allocations,cfg,secmap)
    assert summary["n_sessions"]==80
    assert summary["initial_capital_usd"]==50000
    assert "spy_risk_budget_return_pct" in summary
    assert summary["paper_research_only"] is True
    assert all(trades["paper_only"]) if not trades.empty else True


def test_cross_sectional_sector_and_total_exposure():
    history=fake_market()
    names=("SPY","AAPL","MSFT","JPM","V","XOM")
    panel=prepare_market_panel({s:history[s] for s in names})
    secmap={"AAPL":"Technology","MSFT":"Technology",
            "JPM":"Financials","V":"Financials","XOM":"Energy"}
    cfg=small_config()
    allocations,_=propose_allocations(panel,"cross_sectional_momentum",cfg,secmap)
    assert allocations
    for x in allocations.values():
        assert sum(x.values())<=cfg.max_gross_weight+1e-9
        assert sum(v for k,v in x.items() if secmap[k]=="Technology")<=cfg.max_sector_weight+1e-9


def test_mean_reversion_supported_without_future_data():
    history=fake_market()
    names=["SPY","XLK","XLF","XLE","XLV"]
    panel=prepare_market_panel({s:history[s] for s in names})
    cfg=small_config()
    a,b=propose_allocations(panel,"sector_mean_reversion",cfg,
                            {x:x for x in names if x!="SPY"})
    assert a and len(b)


def test_mutated_future_prices_do_not_change_past_allocations():
    h=fake_market()
    names=["SPY","XLK","XLF","XLE","XLV"]
    cfg=small_config()
    panel=prepare_market_panel({s:h[s] for s in names})
    before,_=propose_allocations(panel,"sector_rotation",cfg,
                                 {s:s for s in names if s!="SPY"})
    modified=fake_market()
    modified["XLK"].loc[328,"Close"]*=1.6
    modified["XLK"].loc[328,"High"]=modified["XLK"].loc[328,"Close"]*1.002
    after,_=propose_allocations(
        prepare_market_panel({s:modified[s] for s in names}),
        "sector_rotation",cfg,{s:s for s in names if s!="SPY"})
    # Signal dated before the modified session cannot change from future data.
    for dt in before:
        if dt<panel.dates[328]:
            assert before[dt]==after[dt]


def test_next_open_execution_and_true_passive_position_drift():
    h=fake_market(90)
    names=["SPY","XLK"]
    panel=prepare_market_panel({s:h[s] for s in names})
    cfg=small_config()
    dt=panel.dates[-12]
    curve,trades,stats=run_portfolio(panel,{dt:{"XLK":0.1}},cfg,{"XLK":"Technology"})
    assert len(trades)==1
    assert pd.Timestamp(trades.iloc[0]["execution_open"])==panel.dates[-11]
    assert pd.Timestamp(curve.iloc[0]["execution_open"])==panel.dates[-11]
    first_r=panel.open.loc[panel.dates[-10],"XLK"]/panel.open.loc[panel.dates[-11],"XLK"]-1
    expected=cfg.initial_capital*(1-(.1*10/10000))*(1+.1*first_r)
    assert curve.iloc[0]["equity_usd"]==pytest.approx(expected)
    assert stats["rebalance_days"]==1


def test_independent_risk_officer_vetoes_invalid_allocations():
    cfg=small_config()
    secmap={"XLK":"Technology","XLF":"Financials","XLE":"Energy"}
    with pytest.raises(ValueError):
        validate_allocation({"XLK":.30},cfg,secmap,set(secmap))
    with pytest.raises(ValueError):
        validate_allocation({"XLK":-.10},cfg,secmap,set(secmap))
    with pytest.raises(ValueError):
        validate_allocation({"XLK":.1,"XLF":.1,"XLE":.1},cfg,
                            {"XLK":"Same","XLF":"Same","XLE":"Same"},set(secmap))
    with pytest.raises(ValueError):
        validate_allocation({"UNKNOWN":.1},cfg,secmap,set(secmap))


def test_drawdown_triggers_permanent_liquidation_at_next_open():
    h=fake_market(90)
    h["XLK"].loc[80:,"Open"]*=.65
    h["XLK"].loc[80:,"Close"]*=.65
    panel=prepare_market_panel({s:h[s] for s in ("SPY","XLK")})
    cfg=small_config(max_drawdown=.02)
    t=panel.dates[78]
    allocations={t:{"XLK":.1},
                 panel.dates[81]:{"XLK":.1},
                 panel.dates[84]:{"XLK":.1}}
    curve,trades,result=run_portfolio(panel,allocations,cfg,{"XLK":"Technology"})
    assert result["risk_stop_triggered"]
    assert (curve["risk_halted"]).any()
    assert (trades["side"]=="SELL").any()


def test_bull_bear_agents_never_authorize_execution():
    debate=investment_debate("AAPL",{"predicted_return":.20})
    assert debate["bull"]["vote"]=="ABSTAIN"
    assert debate["bear"]["vote"]=="OPPOSE"
    assert not debate["trade_execution_allowed"]
    grounded=investment_debate("AAPL",
         {"predicted_return":.03,"out_of_sample_validated":True},
         {"source_urls":["https://www.sec.gov/test"],"narrative_reviewed":False,
          "news_reviewed":False})
    assert grounded["bull"]["vote"]=="SUPPORT_RESEARCH_ONLY"
    assert grounded["bear"]["vote"]=="OPPOSE"
    assert not grounded["investment_committee_authorization"]
