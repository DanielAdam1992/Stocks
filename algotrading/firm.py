"""Deterministic, paper-only investment committee scaffold. No broker execution."""
from dataclasses import dataclass, asdict
from typing import Optional
import json

@dataclass(frozen=True)
class Research:
    ticker: str
    asof: str
    predicted_return: Optional[float]
    financials_reviewed: bool = False
    news_reviewed: bool = False
    evidence: tuple = ()
    red_flags: tuple = ()

@dataclass(frozen=True)
class Decision:
    ticker: str
    action: str
    target_weight: float
    rationale: str
    risk_approved: bool

def committee_decision(research: Research, portfolio_value: float = 50000.0,
                       max_weight: float = 0.10, threshold: float = 0.01) -> Decision:
    """Missing evidence => HOLD; risk officer can veto, never infer approvals."""
    if portfolio_value <= 0 or not (0 < max_weight <= 0.25):
        raise ValueError("Invalid portfolio or allocation limit")
    if research.red_flags:
        return Decision(research.ticker, "HOLD", 0, "Risk veto: "+", ".join(research.red_flags), False)
    if not (research.financials_reviewed and research.news_reviewed and research.evidence):
        return Decision(research.ticker, "HOLD", 0, "Insufficient documented research", False)
    if research.predicted_return is None:
        return Decision(research.ticker, "HOLD", 0, "No validated forecast", False)
    if research.predicted_return <= threshold:
        return Decision(research.ticker, "HOLD", 0, "Forecast below entry threshold", True)
    return Decision(research.ticker, "PROPOSE_BUY", max_weight,
                    "Research gates passed; requires paper execution validation", True)

def write_decision(decision: Decision, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(decision), f, indent=2)
