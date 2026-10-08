"""Narrative candidates and current-report tests use artificial SEC content."""
from datetime import date
from sec_narrative import extract_section_candidates, extract_recent_8k, read_filing_candidates

HTML = """<html>
<head><script>item 1a risk factors Fake risk</script></head>
<body>
<p>Contents: Item 1A. Risk Factors .......... Item 1B</p>
<h2>ITEM 1A. RISK FACTORS</h2>
<p>Actual risk discussion about significant supply dependencies and unexpected demand,
regulatory restrictions, cybersecurity events, interruptions, interest-rate conditions,
and material uncertainty across several operating regions. These statements are
illustrative synthetic text for testing and are not from any real issuer.</p>
<h2>ITEM 1B Unresolved Staff Comments</h2>
<h2>ITEM 7 Management's Discussion and Analysis</h2>
<p>Management reports a shift in revenue mix and constrained cash conversion for
illustrative business segments, while operating costs increased due to overhead,
procurement challenges, inventory timing and additional product development.
This section is only synthetic testing text.</p>
<h2>ITEM 7A Quantitative and Qualitative Disclosures</h2>
</body></html>"""

def test_narrative_sections_extracted_as_unverified():
    sections = extract_section_candidates(HTML, "10-K")
    assert sections["risk_factors"]["status"] == "CANDIDATE_UNVERIFIED"
    assert "significant supply dependencies" in sections["risk_factors"]["excerpt"]
    assert sections["management_discussion"]["status"] == "CANDIDATE_UNVERIFIED"
    assert "Management reports" in sections["management_discussion"]["excerpt"]
    assert "Fake risk" not in sections["risk_factors"]["excerpt"]

def test_missing_section_does_not_invent_analysis():
    sections = extract_section_candidates("<p>No section</p>", "10-Q")
    assert sections["risk_factors"]["status"] == "NOT_FOUND"
    assert sections["risk_factors"]["excerpt"] == ""

def test_recent_8k_obeys_asof_and_filing_cutoff():
    arr = [
        ("8-K", "2026-05-20", "0000320193-26-000003", "evt1.htm"),
        ("8-K", "2026-07-20", "0000320193-26-000004", "evt2.htm"),
        ("8-K", "2026-04-20", "0000320193-26-000005", "evt3.htm"),
        ("10-Q", "2026-05-01", "0000320193-26-000006", "quarter.htm"),
    ]
    s = {"filings": {"recent": {
        "form": [x[0] for x in arr], "filingDate": [x[1] for x in arr],
        "reportDate": ["2026-03-31"]*4, "accessionNumber": [x[2] for x in arr],
        "primaryDocument": [x[3] for x in arr]}}}
    events = extract_recent_8k(s, 320193, date(2026,6,1), "2026-05-01")
    assert len(events) == 1
    assert events[0]["filed"] == "2026-05-20"
    assert events[0]["url"].startswith("https://www.sec.gov/Archives/")

def test_document_reader_does_not_mark_review_complete():
    class Fake:
        def get_filing_html(self, url):
            assert url.startswith("https://www.sec.gov/Archives/")
            return HTML
    filing = {"form":"10-K", "accession":"0000320193-26-000001",
              "primary_document":"form10k.htm"}
    result = read_filing_candidates(Fake(), 320193, filing)
    assert result["narrative_reviewed"] is False
    assert result["sections"]["risk_factors"]["status"] == "CANDIDATE_UNVERIFIED"
