"""
Kalshi API wrapper — direct HTTP with RSA-PSS auth.
Bypasses SDK Pydantic model validation which rejects null fields from the real API.
PAPER TRADING - NO REAL ORDERS
"""

import os
import time
import logging
import requests
from datetime import datetime, timezone
from typing import Optional

from kalshi_python_sync import KalshiAuth

from kalshi_bot.config import (
    KALSHI_API_KEY_ID,
    KALSHI_PRIVATE_KEY_PATH,
    MIN_LIQUIDITY,
    MIN_TIME_TO_CLOSE_HOURS,
    MAX_TIME_TO_CLOSE_HOURS,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


def _load_auth() -> Optional[KalshiAuth]:
    # Read directly from os.environ so Railway vars are never overridden by .env
    api_key_id = os.environ.get("KALSHI_API_KEY_ID", "") or KALSHI_API_KEY_ID
    if not api_key_id:
        logger.warning("No API credentials — running in offline mode")
        return None
    try:
        pem = os.environ.get("KALSHI_PRIVATE_KEY_CONTENT", "")
        if pem:
            pem = pem.replace("\\n", "\n")
        if not pem:
            key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "") or KALSHI_PRIVATE_KEY_PATH
            if key_path:
                pem = open(key_path).read()
        if not pem:
            logger.warning("No API credentials — running in offline mode")
            return None
        return KalshiAuth(api_key_id, pem)
    except Exception as exc:
        logger.warning(f"Could not load Kalshi auth: {exc}")
        return None


def _request(auth: KalshiAuth, method: str, path: str,
             params: dict = None, body: dict = None,
             max_retries: int = 5) -> Optional[dict]:
    url = BASE_URL + path
    for attempt in range(max_retries):
        try:
            headers = auth.create_auth_headers(method, url)
            headers["Content-Type"] = "application/json"
            resp = requests.request(
                method, url, headers=headers,
                params=params, json=body, timeout=15
            )
            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"Rate limited — waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            # Don't retry client errors (4xx) except rate limits (429)
            if e.response is not None and 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                logger.error(f"HTTP error {method} {path}: {e}")
                return None
            if attempt == max_retries - 1:
                logger.error(f"HTTP error {method} {path}: {e}")
                return None
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Request error {method} {path}: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def _parse_market_event(m: dict, category: str = "unknown") -> Optional[dict]:
    """Parse a market nested inside an /events response (dollar-based prices)."""
    ticker = m.get("ticker", "")
    # Skip multivariate parlay markets
    if "KXMVE" in ticker:
        return None

    close_time_str = m.get("close_time") or m.get("expiration_time")
    close_time = None
    if close_time_str:
        try:
            close_time = datetime.fromisoformat(
                str(close_time_str).replace("Z", "+00:00")
            )
        except Exception:
            pass

    now = datetime.now(timezone.utc)
    hours_to_close = (
        (close_time - now).total_seconds() / 3600 if close_time else 999
    )

    # Prices come as dollar strings like "0.4200" → multiply by 100 for cents
    def _d(key: str) -> Optional[int]:
        v = m.get(key)
        return round(float(v) * 100) if v is not None else None

    yes_bid   = _d("yes_bid_dollars")
    yes_ask   = _d("yes_ask_dollars")
    last_p    = _d("last_price_dollars") or _d("previous_price_dollars")
    yes_price = int((yes_bid + yes_ask) / 2) if (yes_bid and yes_ask) else (yes_bid or yes_ask or last_p or 50)

    volume       = float(m.get("volume_fp") or 0)
    volume_24h   = float(m.get("volume_24h_fp") or 0)
    open_interest = float(m.get("open_interest_fp") or 0)

    return {
        "ticker":         ticker,
        "title":          m.get("title", "") or "",
        "yes_price":      int(yes_price),
        "yes_bid":        int(yes_bid) if yes_bid is not None else int(yes_price),
        "yes_ask":        int(yes_ask) if yes_ask is not None else int(yes_price),
        "no_price":       100 - int(yes_price),
        "last_price":     int(last_p) if last_p is not None else int(yes_price),
        "volume":         volume,
        "volume_24h":     volume_24h,
        "open_interest":  open_interest,
        "liquidity":      0.0,
        "close_time":     close_time,
        "hours_to_close": hours_to_close,
        "category":       category or "unknown",
        "status":         m.get("status", "open"),
        "result":         m.get("result"),
        "settled_price":  None,
    }


def _parse_market(m: dict) -> dict:
    close_time_str = m.get("close_time") or m.get("expiration_time")
    close_time = None
    if close_time_str:
        try:
            close_time = datetime.fromisoformat(
                str(close_time_str).replace("Z", "+00:00")
            )
        except Exception:
            pass

    now = datetime.now(timezone.utc)
    hours_to_close = (
        (close_time - now).total_seconds() / 3600
        if close_time else 999
    )

    yes_bid  = m.get("yes_bid") or 0
    yes_ask  = m.get("yes_ask") or 0
    last_price = m.get("last_price") or m.get("yes_bid") or 50
    yes_price = int((yes_bid + yes_ask) / 2) if (yes_bid and yes_ask) else int(last_price)

    return {
        "ticker":         m.get("ticker", ""),
        "title":          m.get("title", "") or m.get("subtitle", "") or "",
        "yes_price":      yes_price,
        "yes_bid":        int(yes_bid) if yes_bid else yes_price,
        "yes_ask":        int(yes_ask) if yes_ask else yes_price,
        "no_price":       100 - yes_price,
        "last_price":     int(last_price),
        "volume":         int(m.get("volume") or 0),
        "volume_24h":     int(m.get("volume_24h") or 0),
        "open_interest":  int(m.get("open_interest") or 0),
        "liquidity":      int(m.get("liquidity") or 0),
        "close_time":     close_time,
        "hours_to_close": hours_to_close,
        "category":       m.get("category") or "unknown",
        "status":         m.get("status", "open"),
        "result":         m.get("result"),          # 'yes'/'no'/None
        "settled_price":  m.get("settled_price"),
    }


class KalshiClient:
    def __init__(self):
        self._auth = _load_auth()
        if self._auth:
            logger.info("Kalshi client ready (live API)")
        else:
            logger.warning("No API credentials — running in offline mode")

    def _is_online(self) -> bool:
        return self._auth is not None

    # ── Market data ────────────────────────────────────────────────

    def get_all_active_markets(self) -> list[dict]:
        """
        Fetch all active markets via the /events endpoint.
        This returns real single-binary markets with live pricing,
        skipping the zero-volume multivariate parlay markets.
        """
        if not self._is_online():
            return []

        markets = []
        cursor = None
        page = 0

        while True:
            params = {
                "status": "open",
                "limit": 200,
                "with_nested_markets": "true",
            }
            if cursor:
                params["cursor"] = cursor

            data = _request(self._auth, "GET", "/events", params=params)
            if not data:
                break

            for ev in data.get("events") or []:
                category = ev.get("category") or "unknown"
                for m in (ev.get("markets") or []):
                    parsed = _parse_market_event(m, category)
                    if parsed:
                        markets.append(parsed)

            cursor = data.get("cursor")
            page += 1

            if not cursor or not data.get("events"):
                break
            if page > 100:
                break

        logger.info(f"Fetched {len(markets)} active markets from events API")
        return markets

    def get_market(self, ticker: str) -> Optional[dict]:
        if not self._is_online():
            return None
        data = _request(self._auth, "GET", f"/markets/{ticker}")
        if data and data.get("market"):
            return _parse_market(data["market"])
        return None

    def get_orderbook(self, ticker: str, depth: int = 10) -> Optional[dict]:
        if not self._is_online():
            return None
        data = _request(self._auth, "GET", f"/markets/{ticker}/orderbook",
                        params={"depth": depth})
        if not data:
            return None
        ob = data.get("orderbook") or {}
        return {
            "yes": ob.get("yes") or [],  # [[price, size], ...]
            "no":  ob.get("no") or [],
        }

    def get_market_history(self, ticker: str, limit: int = 100) -> list[dict]:
        if not self._is_online():
            return []
        data = _request(self._auth, "GET", f"/markets/{ticker}/history",
                        params={"limit": limit})
        if not data:
            return []
        history = data.get("history") or []
        return [
            {
                "yes_price": int(h.get("yes_price") or h.get("price") or 50),
                "timestamp": h.get("ts") or h.get("created_time"),
            }
            for h in history
        ]

    def get_trades(self, ticker: str, limit: int = 100) -> list[dict]:
        """Recent matched trades for a market."""
        if not self._is_online():
            return []
        data = _request(self._auth, "GET", "/markets/trades",
                        params={"ticker": ticker, "limit": limit})
        if not data:
            return []
        return data.get("trades") or []

    def get_portfolio(self) -> dict:
        """Current account balance and positions."""
        if not self._is_online():
            return {}
        data = _request(self._auth, "GET", "/portfolio/balance")
        return data or {}

    def get_positions(self) -> list[dict]:
        if not self._is_online():
            return []
        data = _request(self._auth, "GET", "/portfolio/positions")
        return (data or {}).get("market_positions") or []

    def get_settlements(self, max_pages: int = 20) -> list[dict]:
        """Fetch all recently settled markets (paginated)."""
        if not self._is_online():
            return []
        all_settlements = []
        cursor = None
        for _ in range(max_pages):
            params = {"limit": 200}
            if cursor:
                params["cursor"] = cursor
            data = _request(self._auth, "GET", "/portfolio/settlements", params=params)
            if not data:
                break
            batch = data.get("settlements") or []
            all_settlements.extend(batch)
            cursor = data.get("cursor")
            if not cursor or not batch:
                break
        return all_settlements

    def place_order(self, ticker: str, side: str, size: int, price: int) -> dict:
        # PAPER TRADING - NO REAL ORDERS — logs intent only
        logger.info(
            f"[PAPER ORDER] ticker={ticker} side={side} size={size} price={price}"
        )
        return {
            "status": "paper_filled",
            "ticker": ticker, "side": side,
            "size": size, "price": price,
        }


# Sports markets showed 30.8% WR and -$600 loss in walk-forward backtest — skip entirely
BLOCKED_CATEGORIES = {"Sports"}

def filter_tradeable(markets: list[dict]) -> list[dict]:
    tradeable = []
    for m in markets:
        if m.get("category", "").strip() in BLOCKED_CATEGORIES:
            continue
        vol_ok = (m.get("volume_24h", 0) >= 500 or m.get("volume", 0) >= MIN_LIQUIDITY)
        if not vol_ok:
            continue
        if m["hours_to_close"] < MIN_TIME_TO_CLOSE_HOURS:
            continue
        if m["hours_to_close"] > MAX_TIME_TO_CLOSE_HOURS:
            continue
        if not (3 <= m["yes_price"] <= 97):
            continue
        tradeable.append(m)
    return tradeable
