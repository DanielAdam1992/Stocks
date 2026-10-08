"""Paper broker must never send orders nor permit live-host override."""
import io
import json
import pytest
from urllib.request import Request

from paper_broker_readonly import (
    PAPER_HOST, PaperConnectionError, paper_get,read_paper_status)


class FixtureResponse:
    def __init__(self, payload):
        self._raw=io.BytesIO(json.dumps(payload).encode())
    def __enter__(self):
        return self._raw
    def __exit__(self,*args):
        self._raw.close()


def test_only_get_paper_host_and_no_orders():
    observed=[]
    def fake(req,timeout):
        assert isinstance(req,Request)
        observed.append((req.get_method(),req.full_url))
        assert req.get_method()=="GET"
        assert req.full_url.startswith(PAPER_HOST+"/")
        if req.full_url.endswith("/v2/account"):
            return FixtureResponse({"equity":"50000","cash":"50000",
                                    "status":"ACTIVE","trading_blocked":False,
                                    "account_number":"REDACTED"})
        return FixtureResponse({"is_open":False,"next_open":"fixture"})
    out=paper_get("/v2/clock",key="synthetic",secret="synthetic",opener=fake)
    assert out["is_open"] is False
    assert all(method=="GET" for method,url in observed)
    assert not any("/orders" in url for _,url in observed)


def test_live_host_and_mutations_are_not_configurable():
    with pytest.raises(PaperConnectionError):
        paper_get("/v2/orders",key="test",secret="test",opener=lambda *a:None)
    with pytest.raises(PaperConnectionError):
        paper_get("https://api.alpaca.markets/v2/account",
                  key="test",secret="test",opener=lambda *a:None)
    with pytest.raises(PaperConnectionError):
        paper_get("/v2/account",key="",secret="",opener=lambda *a:None)


def test_health_output_redacts_account_identifiers(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID","secret-example")
    monkeypatch.setenv("APCA_API_SECRET_KEY","secret2-example")
    def fake(req,timeout):
        if req.full_url.endswith("/v2/account"):
            return FixtureResponse({"equity":"50000","cash":"50000",
                                    "status":"ACTIVE","trading_blocked":False,
                                    "account_number":"999999"})
        return FixtureResponse({"is_open":False,"next_open":"tomorrow"})
    out=read_paper_status(opener=fake)
    assert out["orders_submitted"]==0
    assert out["order_api_available"] is False
    assert "999999" not in json.dumps(out)
    assert "secret-example" not in json.dumps(out)
