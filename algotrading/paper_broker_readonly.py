"""Read-only Alpaca *paper* account health check. No order submission code.

The live trading host is not configurable and POST/PATCH/DELETE are unsupported.
Secrets are read from environment and never written to output files or logs.
"""
from __future__ import annotations

import argparse
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PAPER_HOST="https://paper-api.alpaca.markets"
ALLOWED_GET={
    "/v2/account",
    "/v2/clock",
    "/v2/positions",
    "/v2/orders?status=open&limit=100",
}


class PaperConnectionError(RuntimeError):
    pass


def paper_get(path: str, *,key: str, secret: str,
              opener=urlopen,timeout: int=12) -> object:
    if path not in ALLOWED_GET:
        raise PaperConnectionError("Endpoint not allowed: paper GET only")
    if not key or not secret:
        raise PaperConnectionError(
            "Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY paper credentials")
    if timeout<1 or timeout>30:
        raise ValueError("Invalid request timeout")
    req=Request(PAPER_HOST+path,headers={
        "APCA-API-KEY-ID":key,
        "APCA-API-SECRET-KEY":secret,
        "Accept":"application/json",
        "User-Agent":"Trader-Paper-Research/0.1",
    },method="GET")
    if not req.full_url.startswith(PAPER_HOST+"/"):
        raise PaperConnectionError("Only Alpaca paper host permitted")
    try:
        with opener(req,timeout=timeout) as resp:
            obj=json.load(resp)
        return obj
    except (HTTPError,URLError,TimeoutError,ValueError) as exc:
        # Never include keys in messages.
        raise PaperConnectionError(f"Paper connectivity failed: {type(exc).__name__}") from None


def read_paper_status(*,opener=urlopen) -> dict:
    """Readiness information only, with account identifiers redacted."""
    key=os.getenv("APCA_API_KEY_ID","")
    secret=os.getenv("APCA_API_SECRET_KEY","")
    account=paper_get("/v2/account",key=key,secret=secret,opener=opener)
    clock=paper_get("/v2/clock",key=key,secret=secret,opener=opener)
    if not isinstance(account,dict) or not isinstance(clock,dict):
        raise PaperConnectionError("Unexpected response from paper broker")
    return {
        "broker":"Alpaca",
        "endpoint":PAPER_HOST,
        "mode":"PAPER_READ_ONLY",
        "account_status":account.get("status"),
        "trading_blocked":account.get("trading_blocked"),
        "account_equity_usd":account.get("equity"),
        "cash_usd":account.get("cash"),
        "market_open":clock.get("is_open"),
        "next_open":clock.get("next_open"),
        "orders_submitted":0,
        "order_api_available":False,
        "risk_committee_authorized":False,
        "warning":"Connectivity check only; not a $50k deposit or broker trading authorization",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--check",action="store_true",help="Read-only paper account health check")
    args=p.parse_args()
    if not args.check:
        p.error("Pass --check explicitly. This command never places orders.")
    try:
        print(json.dumps(read_paper_status(),indent=2))
    except PaperConnectionError as exc:
        p.error(str(exc))


if __name__=="__main__":
    main()
