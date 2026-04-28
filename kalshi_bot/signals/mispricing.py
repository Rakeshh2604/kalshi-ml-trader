"""
Mispricing signal: detects statistical edge from price history and spread.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

NONE_SIGNAL = {"signal": "NONE", "edge": 0.0, "confidence": 0.0}


def analyze(market: dict) -> dict:
    yes_price = market.get("yes_price", 50)
    yes_bid = market.get("yes_bid", yes_price)
    yes_ask = market.get("yes_ask", yes_price)
    history = market.get("history", [])

    implied_prob = yes_price / 100.0

    # Spread signal
    spread = yes_ask - yes_bid
    spread_signal = spread > 5  # wide spread = potential edge

    # Rolling fair value from history
    if len(history) >= 10:
        prices = [h["yes_price"] for h in history[-20:]]
        fair_value = float(np.mean(prices))
        std = float(np.std(prices))
        deviation = yes_price - fair_value

        if std < 1:
            return NONE_SIGNAL

        z_score = abs(deviation) / std

        if z_score < 1.5:
            return NONE_SIGNAL

        edge = min(abs(deviation) / 100.0, 0.30)
        confidence = min(0.5 + z_score * 0.1, 0.90)

        signal = "NO" if deviation > 0 else "YES"

        # Boost confidence if spread also wide
        if spread_signal:
            confidence = min(confidence + 0.05, 0.95)

        logger.debug(
            f"[Mispricing] {market['ticker']} z={z_score:.2f} edge={edge:.3f} signal={signal}"
        )
        return {"signal": signal, "edge": edge, "confidence": confidence}

    # Fallback: spread-only heuristic
    if spread_signal and 10 <= yes_price <= 90:
        edge = spread / 200.0
        return {"signal": "YES" if implied_prob < 0.5 else "NO", "edge": edge, "confidence": 0.52}

    return NONE_SIGNAL
