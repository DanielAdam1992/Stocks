"""Unit tests use synthetic SEC responses; never imply market performance."""
from datetime import date
import pytest
from sec_analyst import (SECClient, SECError, analyze_company, build_research_packet,
                         latest_financial_filing, metric_from_filing, filing_url)
from firm import Research, committee_decision

CIK = 320193
Q1 = {"form": "10-Q", "filed": "2026-05-01", "end": "2026-03-31",
      "accession": "0000320193-26-000001", "primary_document": "quarterly.htm"}
FUTURE = {"form": "10-Q", "filed": "2026-08-01", "end": "2026-06-30",
          "accession": "0000320193-26-000002", "primary_document": "quarterly2.htm"}
K2025 = {"form": "10-K", "filed": "2025-11-01", "end": "2025-09-30",
         "accession": "0000320193-25-000100", "primary_document": "annual.htm"}

def submissions(filings=(K2025, FUTURE, Q1)):
    return {"filings": {"recent": {
        "form": [f["form"] for f in filings],
        "filingDate": [f["filed"] for f in filings],
        "reportDate": [f["end"] for f in filings],
        "accessionNumber": [f["accession"] for f in filings],
        "primaryDocument": [f["primary_document"] for f in filings]}}}

def fact(filing, value, start=None):
    row = {"accn": filing["accession"], "form": filing["form"],
           "filed": filing["filed"], "end": filing["end"], "val": value}
    if start: row["start"] = start
    return row

def facts():
    def concept(rows):
        return {"units": {"USD": rows}}
    return {"facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": concept([
            fact(Q1, 100, "2026-01-01"), fact(FUTURE, 999999, "2026-04-01")]),
        "NetIncomeLoss": concept([fact(Q1, 20, "2026-01-01")]),
        "Assets": concept([fact(Q1, 500)]),
        "Liabilities": concept([fact(Q1, 300)]),
        "AssetsCurrent": concept([fact(Q1, 300)]),
        "LiabilitiesCurrent": concept([fact(Q1, 100)]),
        "NetCashProvidedByUsedInOperatingActivities": concept([
            fact(Q1, 35, "2026-01-01")]),
    }}}

def report(asof=date(2026, 6, 1), supplied_facts=None):
    return analyze_company("AAPL", CIK, "Apple Inc", submissions(),
                           supplied_facts if supplied_facts is not None else facts(),
                           asof)

def test_asof_excludes_future_filing_and_facts():
    latest = latest_financial_filing(submissions(), date(2026, 6, 1))
    assert latest["accession"] == Q1["accession"]
    r = report()
    assert r.metrics_verified
    assert r.filing["end"] == "2026-03-31"
    assert r.metrics["revenue"]["value"] == 100
    assert "999999" not in repr(r.metrics)
    assert r.source_url == filing_url(CIK, Q1)
    assert any("20.00%" in s for s in r.observations)

def test_periods_do_not_mix():
    f = facts()
    f["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"] = [
        fact(Q1, 1000, "2025-10-01")  # YTD not quarterly
    ]
    assert metric_from_filing(f, "revenue", Q1) is None
    assert not report(supplied_facts=f).metrics_verified

def test_missing_metric_fails_closed():
    f = facts()
    del f["facts"]["us-gaap"]["Assets"]
    r = report(supplied_facts=f)
    assert not r.metrics_verified
    assert any("missing_required_metrics" in b for b in r.blockers)

def test_old_filing_is_blocked():
    old = report(asof=date(2027, 6, 1))
    assert any(b.startswith("filing_stale_") for b in old.blockers)

def test_no_filing_before_asof():
    r = report(asof=date(2025, 1, 1))
    assert not r.metrics_verified
    assert "no_asof_10k_or_10q" in r.blockers

def test_committee_does_not_promote_unaudited_data_to_order():
    r = report()
    packet = build_research_packet(r, predicted_return=0.20)
    assert not packet["financials_reviewed"]
    assert not packet["news_reviewed"]
    assert packet["financial_metrics_verified"]
    allowed = {k: packet[k] for k in Research.__dataclass_fields__}
    decision = committee_decision(Research(**allowed))
    assert decision.action == "HOLD"
    assert not decision.risk_approved

def test_requires_declared_user_agent():
    with pytest.raises(SECError):
        SECClient("anonymous-bot")

def test_ticker_mapping_exact():
    class MockClient(SECClient):
        def get_json(self, url):
            return {"0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."}}
    client = MockClient("Research research@example.com")
    assert client.resolve("aapl") == (320193, "Apple Inc.")
    with pytest.raises(SECError):
        client.resolve("MSFT")

def test_truncated_submissions_fails_closed():
    broken = submissions()
    broken["filings"]["recent"]["form"].pop()
    with pytest.raises(SECError):
        latest_financial_filing(broken, date(2026, 6, 1))
