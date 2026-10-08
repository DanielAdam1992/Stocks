"""SEC EDGAR financial analyst: reproducible, as-of, source-grounded, paper-only.

Run: SEC_USER_AGENT="Trader Research contact@example.com" python algotrading/sec_analyst.py AAPL
All numbers come from SEC companyfacts; they are *not* investment recommendations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SEC_DATA = "https://data.sec.gov"
SEC_WWW = "https://www.sec.gov"
VALID_FORMS = {"10-K", "10-Q"}
DATE_FORMAT = "%Y-%m-%d"

# Concept aliases vary across US-GAAP reporting companies.
METRIC_TAGS = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet"),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
}
INSTANT_METRICS = {"cash", "assets", "liabilities", "current_assets", "current_liabilities"}
REQUIRED_METRICS = {"revenue", "net_income", "assets", "liabilities"}


class SECError(RuntimeError):
    """Controlled ingestion failure; no inferred evidence is produced."""


class SECClient:
    def __init__(self, user_agent: str, min_interval_seconds: float = 0.5, timeout_seconds: int = 20):
        if not user_agent or not re.search(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b", user_agent):
            raise SECError("Set SEC_USER_AGENT with an application name and contact email")
        self.user_agent = user_agent
        self.min_interval = max(min_interval_seconds, 0.2)
        self.timeout = timeout_seconds
        self._last_call = 0.0

    def get_json(self, url: str) -> dict:
        # Do not allow redirects or caller-supplied arbitrary hosts.
        if not (url.startswith(SEC_DATA + "/") or url.startswith(SEC_WWW + "/")):
            raise SECError("SEC API URL must use an official SEC host")
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        req = Request(url, headers={"User-Agent": self.user_agent,
                                    "Accept": "application/json",
                                    "Accept-Encoding": "identity"})
        try:
            self._last_call = time.monotonic()
            with urlopen(req, timeout=self.timeout) as response:
                payload = response.read(30_000_000 + 1)
                if len(payload) > 30_000_000:
                    raise SECError("SEC response exceeds size limit")
                result = json.loads(payload)
                if not isinstance(result, dict):
                    raise SECError("Unexpected SEC JSON structure")
                return result
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            raise SECError(f"SEC request failed: {type(exc).__name__}: {exc}") from exc

    def resolve(self, ticker: str) -> tuple[int, str]:
        symbol = ticker.strip().upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,11}", symbol):
            raise SECError("Invalid US ticker")
        mapping = self.get_json(f"{SEC_WWW}/files/company_tickers.json")
        for item in mapping.values():
            if str(item.get("ticker", "")).upper() == symbol:
                return int(item["cik_str"]), str(item["title"])
        raise SECError(f"No SEC ticker/CIK mapping found for {symbol}")

    def submissions(self, cik: int) -> dict:
        return self.get_json(f"{SEC_DATA}/submissions/CIK{cik:010d}.json")

    def facts(self, cik: int) -> dict:
        return self.get_json(f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik:010d}.json")


def _valid_date(s: str) -> date:
    return datetime.strptime(s, DATE_FORMAT).date()


def latest_financial_filing(submissions: dict, asof: date) -> Optional[dict]:
    recent = submissions.get("filings", {}).get("recent", {})
    keys = ("form", "filingDate", "reportDate", "accessionNumber", "primaryDocument")
    lengths = [len(recent.get(k, [])) for k in keys]
    if not lengths or min(lengths) == 0 or len(set(lengths)) != 1:
        raise SECError("Invalid filings/recent columns in submissions")
    candidates = []
    for i in range(lengths[0]):
        if recent["form"][i] not in VALID_FORMS:
            continue
        try:
            filed = _valid_date(recent["filingDate"][i])
            period_end = _valid_date(recent["reportDate"][i])
        except (ValueError, TypeError):
            continue
        if filed > asof or period_end > asof:
            continue
        accession = str(recent["accessionNumber"][i])
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
            continue
        primary = str(recent["primaryDocument"][i])
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,200}", primary):
            continue
        candidates.append({
            "form": recent["form"][i], "filed": filed.isoformat(),
            "end": period_end.isoformat(), "accession": accession,
            "primary_document": primary})
    if not candidates:
        return None
    return max(candidates, key=lambda x: (x["filed"], x["accession"]))


def filing_url(cik: int, filing: dict) -> str:
    return (f"{SEC_WWW}/Archives/edgar/data/{cik}/"
            f"{filing['accession'].replace('-', '')}/{filing['primary_document']}")


def _match_period(row: dict, filing: dict, instant: bool) -> bool:
    if row.get("accn") != filing["accession"] or row.get("form") != filing["form"]:
        return False
    if row.get("end") != filing["end"] or row.get("filed") != filing["filed"]:
        return False
    if instant:
        return not row.get("start")
    if not row.get("start"):
        return False
    try:
        days = (_valid_date(row["end"]) - _valid_date(row["start"])).days + 1
    except (ValueError, TypeError):
        return False
    # Quarterly cash flow is frequently fiscal-year-to-date, unlike quarterly P&L.
    if filing["form"] == "10-Q":
        if row.get("_metric") == "operating_cash_flow":
            return 65 <= days <= 310
        return 65 <= days <= 115
    return 330 <= days <= 400


def metric_from_filing(facts: dict, metric: str, filing: dict) -> Optional[dict]:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in METRIC_TAGS[metric]:
        units = gaap.get(tag, {}).get("units", {})
        rows = units.get("USD", [])
        matches = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("val"), (int, float)):
                continue
            candidate = dict(row)
            candidate["_metric"] = metric
            if _match_period(candidate, filing, metric in INSTANT_METRICS):
                matches.append(row)
        if matches:
            # Prefer exact period; for cash flow, take longest YTD for the current filing.
            best = max(matches, key=lambda r: (r.get("start", ""), r.get("frame", "")))
            return {"value": best["val"], "unit": "USD", "tag": tag,
                    "start": best.get("start"), "end": best["end"],
                    "filed": best["filed"], "accession": best["accn"],
                    "period_basis": ("instant" if metric in INSTANT_METRICS else
                                     "annual" if filing["form"] == "10-K" else
                                     "year_to_date" if metric == "operating_cash_flow" else "quarter")}
    return None


@dataclass(frozen=True)
class FinancialReport:
    ticker: str
    company: str
    cik: int
    asof: str
    filing: Optional[dict]
    source_url: Optional[str]
    facts_url: str
    metrics: dict
    observations: tuple[str, ...]
    blockers: tuple[str, ...]
    metrics_verified: bool

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_company(ticker: str, cik: int, company: str, submissions: dict,
                    facts: dict, asof: date, max_age_days: int = 210) -> FinancialReport:
    if max_age_days < 1:
        raise ValueError("max_age_days must be positive")
    filing = latest_financial_filing(submissions, asof)
    source_url = filing_url(cik, filing) if filing else None
    facts_url = f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik:010d}.json"
    metrics = {k: metric_from_filing(facts, k, filing) if filing else None for k in METRIC_TAGS}
    blockers = []
    observations = []
    if not filing:
        blockers.append("no_asof_10k_or_10q")
    else:
        age = (asof - _valid_date(filing["filed"])).days
        if age > max_age_days:
            blockers.append(f"filing_stale_{age}_days")
    missing = sorted(k for k in REQUIRED_METRICS if metrics[k] is None)
    if missing:
        blockers.append("missing_required_metrics:" + ",".join(missing))

    rev = metrics["revenue"]
    income = metrics["net_income"]
    if rev and income and rev["value"] > 0 and rev["start"] == income["start"]:
        margin = income["value"] / rev["value"]
        observations.append(f"Reported-period net margin: {margin:.2%}")
    if income and income["value"] < 0:
        observations.append("Reported-period net loss; human review required")
    ca, cl = metrics["current_assets"], metrics["current_liabilities"]
    if ca and cl and cl["value"] > 0:
        observations.append(f"Reported current ratio: {ca['value']/cl['value']:.2f}")
    ocf = metrics["operating_cash_flow"]
    if ocf:
        observations.append(f"Reported operating cash flow ({ocf['period_basis']}): USD {ocf['value']:,.0f}")
    observations.append("Narrative risk factors, earnings call and subsequent 8-Ks NOT reviewed")
    return FinancialReport(
        ticker=ticker.upper(), company=company, cik=cik,
        asof=asof.isoformat(), filing=filing, source_url=source_url,
        facts_url=facts_url, metrics=metrics,
        observations=tuple(observations), blockers=tuple(blockers),
        metrics_verified=not blockers)


def build_research_packet(report: FinancialReport, predicted_return: Optional[float] = None) -> dict:
    """Fail closed: structured facts do not equal a complete financial or news review."""
    return {
        "ticker": report.ticker, "asof": report.asof,
        "predicted_return": predicted_return,
        "financials_reviewed": False,
        "news_reviewed": False,
        "evidence": tuple(u for u in (report.source_url, report.facts_url) if u),
        "red_flags": tuple(report.blockers) + ("filing_narrative_review_pending",),
        "financial_metrics_verified": report.metrics_verified,
    }


def write_report(report: FinancialReport, out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read SEC filing facts (no broker access)")
    parser.add_argument("ticker", help="US equity ticker (e.g. AAPL)")
    parser.add_argument("--asof", default=date.today().isoformat(),
                        help="Information cutoff YYYY-MM-DD; historical backtests must use this")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()
    asof = _valid_date(args.asof)
    if asof > date.today():
        parser.error("--asof must not be in the future")
    try:
        client = SECClient(os.getenv("SEC_USER_AGENT", ""))
        cik, company = client.resolve(args.ticker)
        report = analyze_company(args.ticker, cik, company,
                                 client.submissions(cik), client.facts(cik), asof)
    except SECError as exc:
        parser.error(str(exc))
    out = args.out or f"output/sec_{report.ticker}_{asof.isoformat()}.json"
    write_report(report, out)
    print(json.dumps({
        "ticker": report.ticker, "filing": report.filing,
        "source": report.source_url, "metrics_verified": report.metrics_verified,
        "blockers": report.blockers, "observations": report.observations,
        "saved": out}, indent=2))


if __name__ == "__main__":
    main()
