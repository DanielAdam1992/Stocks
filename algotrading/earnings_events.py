"""Point-in-time earnings-event research, exclusively sourced historical fixtures.

No historical earnings surprises are inferred from price movements, and
no events are backdated from later web results. A separately licensed
timestamped vendor feed is required for real event-strategy results.
"""
from __future__ import annotations

from dataclasses import dataclass,asdict
from datetime import datetime, date, time, timedelta
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

TZ=ZoneInfo("America/New_York")


@dataclass(frozen=True)
class EarningsEvent:
    ticker: str
    announced_at_utc: str
    available_at_utc: str
    event_type: str
    surprise_pct: float
    primary_source_url: str
    provider: str
    provider_record_id: str

    def validate(self):
        event_time=pd.Timestamp(self.announced_at_utc)
        available=pd.Timestamp(self.available_at_utc)
        if event_time.tzinfo is None or available.tzinfo is None:
            raise ValueError("Point-in-time earnings data require timezone-aware timestamps")
        if available < event_time:
            raise ValueError("Cannot observe an event before its announcement")
        if self.event_type!="earnings_release":
            raise ValueError("Unsupported event category")
        if not self.ticker or not self.provider or not self.provider_record_id:
            raise ValueError("Missing identifying event provenance")
        if not self.primary_source_url.startswith("https://"):
            raise ValueError("Sourced URL required")
        if pd.isna(self.surprise_pct) or not -1 <= float(self.surprise_pct)<=10:
            raise ValueError("Invalid signed fractional earnings surprise")


def decision_cutoff_utc(market_day: pd.Timestamp) -> pd.Timestamp:
    """Conservative next-open signal: public information by 23:59 ET."""
    day=pd.Timestamp(market_day).date()
    local=datetime.combine(day,time(23,59),tzinfo=TZ)
    return pd.Timestamp(local).tz_convert("UTC")


def asof_earnings(events:Iterable[EarningsEvent],market_date:pd.Timestamp,
                  *,max_age_calendar_days:int=7) -> list[EarningsEvent]:
    if max_age_calendar_days<1:
        raise ValueError("max_age_calendar_days must be positive")
    cutoff=decision_cutoff_utc(market_date)
    selected=[]
    for e in events:
        e.validate()
        release=pd.Timestamp(e.announced_at_utc).tz_convert("UTC")
        available=pd.Timestamp(e.available_at_utc).tz_convert("UTC")
        if available>cutoff or release>cutoff:
            continue
        if cutoff-available>pd.Timedelta(days=max_age_calendar_days):
            continue
        selected.append(e)
    return sorted(selected,key=lambda x:x.available_at_utc,reverse=True)


def event_candidate_notes(events:Iterable[EarningsEvent],
                          market_date:pd.Timestamp,
                          surprise_threshold:float=.05) -> dict:
    eligible=asof_earnings(events,market_date)
    qualified=[e for e in eligible if e.surprise_pct>=surprise_threshold]
    return {
        "asof_market_date":str(pd.Timestamp(market_date).date()),
        "time_cutoff_utc":str(decision_cutoff_utc(market_date)),
        "source_record_count":len(eligible),
        "positive_surprise_candidates":[asdict(e) for e in qualified],
        "paper_research_only":True,
        "trade_authorized":False,
        "warning":"One-time event screen, not an executed or empirically validated earnings strategy",
    }
