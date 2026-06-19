"""
Live CSE market-data helpers (read-only).
Separate from pipeline/cse_fetcher.py, which handles financial-report discovery.

This is an unofficial, reverse-engineered API (see
https://github.com/GH0STH4CKER/Colombo-Stock-Exchange-CSE-API-Documentation) — response
shapes aren't guaranteed, so parsing here is defensive about wrapper-key names.
"""
import logging

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://www.cse.lk/api"


def _extract_list(payload) -> list:
    """CSE endpoints wrap their array in an undocumented (sometimes typo'd) key."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list):
                return v
    return []


def get_live_prices() -> dict[str, float]:
    """POST tradeSummary -> {base_symbol: price} for every listed security.
    (todaySharePrice was tried first but only ever returns a ~10-row snapshot of
    recent trades regardless of paging params; tradeSummary covers the full
    ~280-security market.)
    CSE symbols carry a board suffix (e.g. "LOLC.N0000") — stripped to match the
    bare tickers ("LOLC") used elsewhere in this app.
    """
    try:
        resp = requests.post(f"{BASE_URL}/tradeSummary", data={}, timeout=30)
        resp.raise_for_status()
        items = _extract_list(resp.json())
        prices = {}
        for item in items:
            symbol = (item.get("symbol") or "").upper()
            price = item.get("price") or item.get("closingPrice")
            if symbol and price is not None:
                base_symbol = symbol.split(".")[0]
                prices[base_symbol] = float(price)
        log.info("Fetched %d live prices from CSE", len(prices))
        return prices
    except Exception as e:
        log.error("CSE live price fetch failed: %s", e)
        return {}
