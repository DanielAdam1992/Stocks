"""Quant research tests use synthetic prices, never pretend these are market returns."""
import numpy as np
import pandas as pd
import pytest

from quant_research import (WalkForwardConfig, make_features, walk_forward,
                            performance, choose_model, model_factory)
from sector_runner import SECTOR_ETFS, BENCHMARK


def synthetic_prices(n=870, seed=73):
    rng=np.random.default_rng(seed)
    daily=rng.normal(0.0002,0.011,n)
    close=100*np.exp(np.cumsum(daily))
    op=np.roll(close,1)
    op[0]=100
    op=op*(1+rng.normal(0,.001,n))
    high=np.maximum(close,op)*1.004
    low=np.minimum(close,op)*.996
    return pd.DataFrame({"Date":pd.bdate_range("2021-01-01",periods=n),
                         "Open":op,"High":high,"Low":low,
                         "Close":close,"Volume":rng.integers(100000,2000000,size=n)})


def test_next_open_label_exact_and_no_target_imputation():
    raw=synthetic_prices(100)
    df=make_features(raw)
    # First output index corresponds to raw row 60, last two labels excluded.
    original=raw.set_index("Date")
    for row in df.iloc[[0,-1]].itertuples():
        pos=raw.index[raw["Date"]==row.Date][0]
        actual=raw.loc[pos+2,"Open"]/raw.loc[pos+1,"Open"]-1
        assert row.target_return == pytest.approx(actual)
    assert df["target_return"].notna().all()
    assert len(df)==100-60-2


def test_future_price_change_cannot_alter_past_features():
    raw=synthetic_prices(180)
    a=make_features(raw)
    changed=raw.copy()
    idx=175
    changed.loc[idx,"Open"]*=1.10
    changed.loc[idx,"Close"]*=1.10
    changed.loc[idx,"High"]=max(changed.loc[idx,["Open","Close"]])*1.004
    changed.loc[idx,"Low"]=min(changed.loc[idx,["Open","Close"]])*.996
    b=make_features(changed)
    cols=["Date","momentum_5d","return_1d","volume_ratio_20d"]
    pd.testing.assert_frame_equal(a[a["Date"]<raw.loc[idx-2,"Date"]][cols].reset_index(drop=True),
                                  b[b["Date"]<raw.loc[idx-2,"Date"]][cols].reset_index(drop=True))


def test_walk_forward_predictions_no_future_train_rows():
    c=WalkForwardConfig(min_train=110,validation_size=40,test_size=50,retrain_every=25)
    preds,choices=walk_forward(synthetic_prices(280),c)
    assert len(preds)==50
    assert all(pd.Timestamp(x["train_end_date"])<pd.Timestamp(x["test_first_date"]) for x in choices)
    assert preds["predicted_return"].notna().all()
    results,summary=performance(preds,c)
    assert len(results)==50
    assert summary["initial_cash_usd"]==50000
    assert summary["assumed_one_way_bps"]==10
    assert isinstance(summary["passes_research_screen"],bool)


def test_no_pretrained_price_forecast_is_sneaked_into_model_selection():
    features=make_features(synthetic_prices(240))
    best,rank=choose_model(features.iloc[:200].reset_index(drop=True),40)
    assert best["model"] in ("zero","ridge","histgb")
    assert all(x["validation_rmse"]>=0 for x in rank)


def test_position_costs_and_zero_signal():
    raw=make_features(synthetic_prices(160)).tail(20).reset_index(drop=True)
    cfg=WalkForwardConfig(min_train=50,validation_size=20,test_size=20,
                          retrain_every=10)
    raw["predicted_return"]=0.0
    _,summary=performance(raw,cfg)
    assert summary["ending_equity_usd"]==50000
    assert summary["n_entries"]==0
    raw["predicted_return"]=0.10
    _,summary=performance(raw,cfg)
    assert summary["n_entries"]==1
    assert summary["n_rebalances"]==1


def test_parameter_validation_and_missing_prices():
    cfg=WalkForwardConfig(max_weight=0.50)
    with pytest.raises(ValueError):
        cfg.validate()
    p=synthetic_prices(120)
    p.loc[5,"Open"]=np.nan
    with pytest.raises(ValueError):
        make_features(p)
    with pytest.raises(ValueError):
        model_factory("invalid",{})


def test_sector_universe_is_complete_and_distinct():
    assert len(SECTOR_ETFS)==11
    assert len(set(SECTOR_ETFS))==11
    assert BENCHMARK["SPY"]=="S&P 500 Benchmark"
