"""Run the financial analyst and CIO committee. Never send orders to a broker."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import date, datetime
import json
import os
from pathlib import Path

from firm import Research, committee_decision
from sec_analyst import SECClient, SECError, analyze_company, build_research_packet
from sec_narrative import read_filing_candidates, extract_recent_8k


def prepare_committee_record(ticker: str, asof: date, client: SECClient) -> dict:
    cik, name = client.resolve(ticker)
    submissions = client.submissions(cik)
    financial_report = analyze_company(
        ticker, cik, name, submissions, client.facts(cik), asof)
    narrative = (read_filing_candidates(client, cik, financial_report.filing)
                 if financial_report.filing else None)
    recent_8k = extract_recent_8k(
        submissions, cik, asof,
        financial_report.filing["filed"] if financial_report.filing else None)
    source_packet = build_research_packet(financial_report)
    if recent_8k:
        source_packet["red_flags"] += ("unreviewed_8k_current_reports",)
    data = {key: source_packet[key] for key in Research.__dataclass_fields__}
    decision = committee_decision(Research(**data), portfolio_value=50000.0)
    return {
        "mode": "PAPER_RESEARCH_ONLY",
        "portfolio_usd": 50000.0,
        "ticker": ticker.upper(),
        "asof": asof.isoformat(),
        "financial_report": financial_report.to_dict(),
        "filing_narrative": narrative,
        "subsequent_8k_reports": recent_8k,
        "research_packet": source_packet,
        "investment_committee": asdict(decision),
        "agent_status": {
            "fundamental_analyst": "FACTS_VERIFIED" if financial_report.metrics_verified else "DATA_BLOCKED",
            "filing_reader": "EXCERPTS_NEED_VERIFICATION" if narrative else "NO_FILING",
            "current_report_monitor": "EVENTS_NEED_REVIEW" if recent_8k else "NO_SUBSEQUENT_8K",
            "market_news_analyst": "NOT_IMPLEMENTED",
            "quant_researcher": "NOT_CONNECTED_TO_VALIDATED_MODEL",
            "portfolio_manager": "NO_APPROVED_TRADE",
            "risk_officer": "VETO" if not decision.risk_approved else "RESEARCH_GATE_PASSED",
            "cio": decision.action,
            "execution_agent": "DISABLED",
        },
        "execution": {"enabled": False, "reason": "No validated strategy, no broker connected"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial analyst -> CIO committee, research only")
    parser.add_argument("ticker")
    parser.add_argument("--asof", default=date.today().isoformat())
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    try:
        asof = datetime.strptime(args.asof, "%Y-%m-%d").date()
        if asof > date.today():
            parser.error("--asof must not be in the future")
        client = SECClient(os.getenv("SEC_USER_AGENT", ""))
        record = prepare_committee_record(args.ticker, asof, client)
    except SECError as exc:
        parser.error(str(exc))
    path = Path(args.out or f"output/committee_{args.ticker.upper()}_{asof.isoformat()}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({
        "ticker": record["ticker"],
        "financials_verified": record["financial_report"]["metrics_verified"],
        "decision": record["investment_committee"]["action"],
        "reason": record["investment_committee"]["rationale"],
        "record": str(path),
        "broker_order": "DISABLED",
    }, indent=2))


if __name__ == "__main__":
    main()
