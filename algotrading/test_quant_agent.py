"""Synthetic quant agent integration tests. No orders, no live market data."""
from datetime import date
import pytest
import quant_agent
from quant_agent import asof_signal, run_agent
from test_quant_research import synthetic_prices


def test_latest_unlabeled_row_is_usable_but_unresolved_label_is_not():
    raw=synthetic_prices(320)
    result=asof_signal(raw,"AAPL",min_train=130,validation_size=40)
    assert result["forecast_asof_market_close"]==raw.iloc[-1]["Date"].date().isoformat()
    assert result["train_last_feature_date"]==raw.iloc[-3]["Date"].date().isoformat()
    assert result["approval_eligible"] is False
    assert result["execution_allowed"] is False
    assert result["predicted_return"]==pytest.approx(float(result["predicted_return"]))


def test_data_after_cutoff_rejected():
    raw=synthetic_prices(320)
    with pytest.raises(ValueError):
        asof_signal(raw,"AAPL",asof=date(2021,1,1),
                    min_train=130,validation_size=40)


def test_cio_rejects_quant_only_suggestion(monkeypatch):
    raw=synthetic_prices(800)
    monkeypatch.setattr(quant_agent,"load_prices",lambda *a,**k:raw)
    result=run_agent("AAPL")
    assert result["committee"]["action"]=="HOLD"
    assert result["committee"]["risk_approved"] is False
    assert result["broker_orders"]==[]
    assert result["quant_research"]["approval_eligible"] is False
