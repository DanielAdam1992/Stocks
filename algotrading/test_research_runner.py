"""End-to-end (synthetic) financial -> narrative -> CIO -> risk gate."""
from datetime import date
from research_runner import prepare_committee_record

class FakeSEC:
    def resolve(self, ticker):
        return 320193, "Synthetic Company"
    def submissions(self, cik):
        return {"filings":{"recent":{
            "form":["8-K","10-Q"],
            "filingDate":["2026-05-10","2026-05-01"],
            "reportDate":["2026-05-09","2026-03-31"],
            "accessionNumber":["0000320193-26-000002","0000320193-26-000001"],
            "primaryDocument":["news.htm","quarter.htm"]}}}
    def facts(self, cik):
        # Missing facts should never result in a trade proposal.
        return {"facts":{"us-gaap":{}}}
    def get_filing_html(self, url):
        return "<p>Item 2 Management's Discussion and Analysis.</p>" + ("Cash flow constraints. "*100) + "<p>Item 3</p>"

def test_institutional_pipeline_never_trades_on_partial_data():
    result = prepare_committee_record("AAPL", date(2026, 6, 1), FakeSEC())
    assert result["financial_report"]["metrics_verified"] is False
    assert result["filing_narrative"]["narrative_reviewed"] is False
    assert result["subsequent_8k_reports"][0]["form"] == "8-K"
    assert result["agent_status"]["current_report_monitor"] == "EVENTS_NEED_REVIEW"
    assert result["agent_status"]["execution_agent"] == "DISABLED"
    assert result["investment_committee"]["action"] == "HOLD"
    assert result["execution"]["enabled"] is False
    assert "unreviewed_8k_current_reports" in result["research_packet"]["red_flags"]
