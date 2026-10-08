from firm import Research, committee_decision

def test_missing_research_blocks_trade():
    r = Research("AAPL","2026-10-08",0.08)
    assert committee_decision(r).action == "HOLD"

def test_risk_veto_overrides_signal():
    r = Research("AAPL","2026-10-08",0.08,True,True,("filing",),("stale financials",))
    d = committee_decision(r)
    assert d.action == "HOLD" and not d.risk_approved

def test_approved_research_only_proposes():
    r = Research("AAPL","2026-10-08",0.08,True,True,("filing",))
    d = committee_decision(r)
    assert d.action == "PROPOSE_BUY" and d.target_weight <= .10
