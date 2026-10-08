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


def prepare_committee_record(ticker: str, asof: date, client: SECClient) -> dict:
    cik, name = client.resolve(ticker)
    financial_report = analyze_company(
        ticker, cik, name, client.submissions(cik), client.facts(cik), asof)
    source_packet = build_research_packet(financial_report)
    data = {key: source_packet[key] for key in Research.__dataclass_fields__}
    decision = committee_decision(Research(**data), portfolio_value=50000.0)
    return {
        "mode": "PAPER_RESEARCH_ONLY",
        "portfolio_usd": 50000.0,
        "ticker": ticker.upper(),
        "asof": asof.isoformat(),
        "financial_report": financial_report.to_dict(),
        "research_packet": source_packet,
        "investment_committee": asdict(decision),
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
