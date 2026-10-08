"""Research-agent adversarial memos, not autonomous LLM investment approvals.

Sources are explicit. Bull/bear roles may challenge each other but can never
turn missing facts or statistical backtest performance into real order authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Mapping


@dataclass(frozen=True)
class AgentArgument:
    role: str
    thesis: str
    evidence: tuple[str,...]
    unresolved: tuple[str,...]
    vote: str


def _valid_urls(items) -> tuple[str,...]:
    if not isinstance(items,(list,tuple)):
        return ()
    return tuple(x for x in items if isinstance(x,str)
                 and x.startswith("https://") and len(x)<500)


def bull_research(quant: Mapping, financial: Mapping|None=None) -> AgentArgument:
    """Support only claims grounded in supplied numbers and sources."""
    urls=_valid_urls((financial or {}).get("source_urls",[]))
    pred=quant.get("predicted_return")
    proven=bool(quant.get("out_of_sample_validated",False))
    if not isinstance(pred,(int,float)) or not proven or not urls:
        return AgentArgument("bull_analyst","Insufficient verified positive thesis",
                             urls,("Validated quant forecast and sourced fundamentals required",),
                             "ABSTAIN")
    if pred<=0:
        return AgentArgument("bull_analyst","Forecast does not support long exposure",
                             urls,(),"OPPOSE")
    return AgentArgument("bull_analyst",
                         f"Validated forecast suggests {pred:.3%} expected move; "
                         "hypothesis still requires downside testing",
                         urls,("Broker costs and qualitative disclosures need independent review",),
                         "SUPPORT_RESEARCH_ONLY")


def bear_research(quant: Mapping, financial: Mapping|None=None) -> AgentArgument:
    source_urls=_valid_urls((financial or {}).get("source_urls",[]))
    warnings=[]
    if not quant.get("out_of_sample_validated",False):
        warnings.append("Forecast lacks verified independent out-of-sample support")
    if not (financial or {}).get("narrative_reviewed",False):
        warnings.append("SEC filing narrative has not been validated")
    if not (financial or {}).get("news_reviewed",False):
        warnings.append("Recent news and material events not reviewed")
    if (financial or {}).get("red_flags"):
        warnings.extend(str(s) for s in financial["red_flags"])
    return AgentArgument(
        "bear_analyst",
        "Challenge trade thesis: risk is unresolved" if warnings else
        "No decisive downside evidence provided; uncertainty remains",
        source_urls,tuple(warnings),"OPPOSE" if warnings else "ABSTAIN")


def investment_debate(ticker: str, quant: Mapping, financial: Mapping|None=None) -> dict:
    bull=bull_research(quant,financial)
    bear=bear_research(quant,financial)
    return {
        "ticker":ticker.upper(),
        "created_at":datetime.now(timezone.utc).isoformat(),
        "bull":asdict(bull),
        "bear":asdict(bear),
        "investment_committee_authorization":False,
        "trade_execution_allowed":False,
        "next_step":"Complete point-in-time fundamental and market review and independent risk approval",
        "note":"Deterministic adversarial research records; no LLM debate or discretionary order authority",
    }
