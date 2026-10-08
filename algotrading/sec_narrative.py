"""Filing narrative *candidate* extractor and SEC 8-K event tracker.

No language model, conclusion or automated approval: extracted sections need review.
"""
from __future__ import annotations

from datetime import date
from html.parser import HTMLParser
import re
from typing import Any

from sec_analyst import SECClient, SECError, filing_url


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style", "noscript", "svg", "ix:header"}:
            self._skip += 1
        if not self._skip and tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "ix:header"} and self._skip:
            self._skip -= 1
        if not self._skip and tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


# Candidate patterns match the section headings in most 10-K / 10-Q reports.
# An occurrence in a table of contents or body cross-reference is possible.
PATTERNS = {
    "10-K": {
        "risk_factors": (r"\bitem\s+1a\s*[.:\-]?\s*risk\s+factors\b",
                         r"\bitem\s+1b\b|\bitem\s+2\b"),
        "management_discussion": (r"\bitem\s+7\s*[.:\-]?\s*management.s\s+discussion\b",
                                  r"\bitem\s+7a\b|\bitem\s+8\b"),
    },
    "10-Q": {
        "risk_factors": (r"\bitem\s+1a\s*[.:\-]?\s*risk\s+factors\b",
                         r"\bitem\s+2\b|\bitem\s+3\b"),
        "management_discussion": (r"\bitem\s+2\s*[.:\-]?\s*management.s\s+discussion\b",
                                  r"\bitem\s+3\b|\bitem\s+4\b"),
    }
}


def extract_section_candidates(html: str, form: str, max_excerpt: int = 3000) -> dict:
    if form not in PATTERNS:
        raise ValueError("Only 10-K or 10-Q documents supported")
    parser = _PlainText()
    parser.feed(html)
    text = parser.text()
    results = {}
    for section, (start_regex, end_regex) in PATTERNS[form].items():
        candidates = []
        for start in re.finditer(start_regex, text, re.IGNORECASE):
            following = text[start.end():]
            end = re.search(end_regex, following, re.IGNORECASE)
            snippet = following[:end.start()] if end else following[:15000]
            snippet = snippet.strip()
            if len(snippet) >= 120:
                candidates.append(snippet)
        if candidates:
            longest = max(candidates, key=len)
            results[section] = {
                "status": "CANDIDATE_UNVERIFIED",
                "excerpt": longest[:max_excerpt],
                "full_candidate_chars": len(longest),
                "truncated": len(longest) > max_excerpt,
            }
        else:
            results[section] = {
                "status": "NOT_FOUND",
                "excerpt": "",
                "full_candidate_chars": 0,
                "truncated": False,
            }
    return results


def extract_recent_8k(submissions: dict, cik: int, asof: date,
                      since_filed: str | None, limit: int = 10) -> list[dict]:
    """Surface current reports filed after a financial statement; no conclusions."""
    recent = submissions.get("filings", {}).get("recent", {})
    fields = ("form", "filingDate", "reportDate", "accessionNumber", "primaryDocument")
    counts = [len(recent.get(field, [])) for field in fields]
    if not counts or len(set(counts)) != 1:
        raise SECError("Malformed SEC submissions for 8-K extraction")
    events = []
    for i in range(counts[0]):
        if recent["form"][i] not in ("8-K", "8-K/A"):
            continue
        filed = recent["filingDate"][i]
        if not isinstance(filed, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", filed):
            continue
        if filed > asof.isoformat() or (since_filed and filed <= since_filed):
            continue
        accession = str(recent["accessionNumber"][i])
        primary = str(recent["primaryDocument"][i])
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
            continue
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,200}", primary):
            continue
        filing = {"accession": accession, "primary_document": primary}
        events.append({"form": recent["form"][i], "filed": filed,
                       "accession": accession, "url": filing_url(cik, filing)})
    return sorted(events, key=lambda event: (event["filed"], event["accession"]), reverse=True)[:limit]


def read_filing_candidates(client: SECClient, cik: int, filing: dict) -> dict:
    url = filing_url(cik, filing)
    html = client.get_filing_html(url)
    sections = extract_section_candidates(html, filing["form"])
    return {
        "document_url": url,
        "form": filing["form"],
        "sections": sections,
        "narrative_reviewed": False,
        "warning": "Heuristic extracts may be headings, table of contents or partial text; verify against filing before reliance",
    }
