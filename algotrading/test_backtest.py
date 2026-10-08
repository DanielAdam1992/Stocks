import pandas as pd
import pytest
from backtest import backtest

def sample(preds=(0.01,0.01,0.01,0.01)):
    return pd.DataFrame({"Date":pd.date_range("2024-01-01",periods=4),"Close":[100,110,100,120],"predicted_return":preds})

def test_no_same_day_signal_execution():
    df, stats = backtest(sample())
    assert df["weight"].iloc[0] == 0
    assert df["weight"].iloc[1] == 0.25
    assert stats["starting_cash"] == 50000

def test_no_signal_means_flat():
    df, stats = backtest(sample((0,0,0,0)))
    assert stats["ending_equity"] == 50000

def test_invalid_data_rejected():
    with pytest.raises(ValueError):
        backtest(sample().assign(Close=[100,-1,100,120]))
