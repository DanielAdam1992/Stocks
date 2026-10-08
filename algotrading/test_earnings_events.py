"""Point-in-time event research synthetic tests; no real earnings claims."""
import pandas as pd
import pytest

from earnings_events import EarningsEvent,asof_earnings,event_candidate_notes,decision_cutoff_utc


def event(*,available="2026-10-07T19:30:00+00:00",announced="2026-10-07T19:00:00+00:00",
          surprise=0.12):
    return EarningsEvent(
        ticker="AAPL",announced_at_utc=announced,available_at_utc=available,
        event_type="earnings_release",surprise_pct=surprise,
        primary_source_url="https://www.sec.gov/Archives/edgar/example",
        provider="synthetic",provider_record_id="abc")


def test_source_data_publication_cutoff_no_lookahead():
    # Signal at end of 2026-10-07 New York time; tomorrow's surprise is illegal.
    past=event()
    future=event(announced="2026-10-08T10:00:00+00:00",
                 available="2026-10-08T10:05:00+00:00")
    a=asof_earnings([past,future],pd.Timestamp("2026-10-07"))
    assert a==[past]
    assert decision_cutoff_utc(pd.Timestamp("2026-10-07"))<pd.Timestamp(
        "2026-10-08T10:05:00+00:00")


def test_vendor_record_delayed_does_not_backdate():
    late=event(announced="2026-10-06T20:00:00+00:00",
               available="2026-10-09T14:00:00+00:00")
    assert asof_earnings([late],pd.Timestamp("2026-10-07"))==[]


def test_sourced_earnings_are_only_research_candidates():
    d=event_candidate_notes([event()],pd.Timestamp("2026-10-07"))
    assert len(d["positive_surprise_candidates"])==1
    assert d["paper_research_only"] is True
    assert d["trade_authorized"] is False


def test_unprovenanceable_events_are_rejected():
    e=event()
    bad=EarningsEvent(**{**e.__dict__,"primary_source_url":"unknown"})
    with pytest.raises(ValueError):
        asof_earnings([bad],pd.Timestamp("2026-10-07"))
    bad=EarningsEvent(**{**e.__dict__,"available_at_utc":"2026-10-07"})
    with pytest.raises(ValueError):
        asof_earnings([bad],pd.Timestamp("2026-10-07"))
