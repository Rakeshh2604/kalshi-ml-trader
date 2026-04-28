"""
Order book analysis signals: imbalance, microprice, depth skew.

In live mode uses real orderbook data from client.
In backtest mode derives synthetic signals from price/spread data.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

NONE_SIGNAL = {"signal": "NONE", "edge": 0.0, "confidence": 0.0}


def _microprice(yes_bid: float, yes_ask: float,
                bid_size: float, ask_size: float) -> float:
    """
    Microprice: size-weighted mid between bid and ask.
    More informative than simple mid when sizes are unequal.
    """
    total = bid_size + ask_size
    if total == 0:
        return (yes_bid + yes_ask) / 2.0
    return (yes_ask * bid_size + yes_bid * ask_size) / total


def _imbalance(bid_size: float, ask_size: float) -> float:
    """
    Order imbalance ratio ∈ [-1, 1].
    +1 = all bids (buying pressure → YES)
    -1 = all asks (selling pressure → NO)
    """
    total = bid_size + ask_size
    if total == 0:
        return 0.0
    return (bid_size - ask_size) / total


def _depth_pressure(orderbook: dict) -> tuple[float, float]:
    """
    Sum first N levels of YES and NO order books.
    Returns (yes_depth_total, no_depth_total).
    """
    yes_levels = orderbook.get("yes", []) or []
    no_levels = orderbook.get("no", []) or []

    yes_depth = sum(
        level[1] if isinstance(level, (list, tuple)) and len(level) > 1 else 0
        for level in yes_levels[:5]
    )
    no_depth = sum(
        level[1] if isinstance(level, (list, tuple)) and len(level) > 1 else 0
        for level in no_levels[:5]
    )
    return float(yes_depth), float(no_depth)


def _synthetic_orderbook(market: dict) -> dict:
    """
    Derive synthetic order book features from spread + momentum
    when real book data is unavailable (backtest / API offline).
    """
    yes_price = market.get("yes_price", 50)
    yes_bid = market.get("yes_bid", yes_price - 1)
    yes_ask = market.get("yes_ask", yes_price + 1)
    momentum = market.get("price_momentum", 0.0)
    volume = market.get("volume", 5000)

    # Simulate size asymmetry correlated with momentum
    base_size = max(volume / 100, 10)
    momentum_bias = np.clip(momentum / 20.0, -0.5, 0.5)

    bid_size = base_size * (1.0 + momentum_bias)
    ask_size = base_size * (1.0 - momentum_bias)

    spread = yes_ask - yes_bid

    return {
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "spread": spread,
        "microprice": _microprice(yes_bid, yes_ask, bid_size, ask_size),
        "imbalance": _imbalance(bid_size, ask_size),
    }


def extract_features(market: dict, orderbook: dict | None = None) -> dict:
    """Extract numeric order book features for ML model."""
    if orderbook and (orderbook.get("yes") or orderbook.get("no")):
        yes_levels = orderbook.get("yes", [])
        no_levels = orderbook.get("no", [])
        yes_bid = yes_levels[0][0] if yes_levels else market.get("yes_bid", 49)
        yes_ask = no_levels[0][0] if no_levels else market.get("yes_ask", 51)
        bid_size = yes_levels[0][1] if yes_levels and len(yes_levels[0]) > 1 else 50
        ask_size = no_levels[0][1] if no_levels and len(no_levels[0]) > 1 else 50
        yes_depth, no_depth = _depth_pressure(orderbook)
    else:
        ob = _synthetic_orderbook(market)
        yes_bid = ob["yes_bid"]
        yes_ask = ob["yes_ask"]
        bid_size = ob["bid_size"]
        ask_size = ob["ask_size"]
        yes_depth = bid_size * 3
        no_depth = ask_size * 3

    mid = (yes_bid + yes_ask) / 2.0
    micro = _microprice(yes_bid, yes_ask, bid_size, ask_size)
    imbalance = _imbalance(bid_size, ask_size)
    depth_ratio = yes_depth / (no_depth + 1e-6)

    return {
        "imbalance": imbalance,
        "microprice_vs_mid": micro - mid,
        "depth_ratio": np.log1p(depth_ratio),
        "spread_cents": yes_ask - yes_bid,
        "relative_spread": (yes_ask - yes_bid) / (mid + 1e-6),
    }


def analyze(market: dict, orderbook: dict | None = None) -> dict:
    """Generate directional signal from order book data."""
    ob = _synthetic_orderbook(market) if not orderbook else None

    if ob is None and orderbook:
        yes_levels = orderbook.get("yes", [])
        no_levels = orderbook.get("no", [])
        yes_bid = yes_levels[0][0] if yes_levels else market.get("yes_bid", 49)
        yes_ask = no_levels[0][0] if no_levels else market.get("yes_ask", 51)
        bid_size = yes_levels[0][1] if yes_levels and len(yes_levels[0]) > 1 else 50
        ask_size = no_levels[0][1] if no_levels and len(no_levels[0]) > 1 else 50
        yes_depth, no_depth = _depth_pressure(orderbook)
    else:
        assert ob is not None
        yes_bid = ob["yes_bid"]
        yes_ask = ob["yes_ask"]
        bid_size = ob["bid_size"]
        ask_size = ob["ask_size"]
        yes_depth = bid_size * 3
        no_depth = ask_size * 3

    imbalance = _imbalance(bid_size, ask_size)
    mid = (yes_bid + yes_ask) / 2.0
    micro = _microprice(yes_bid, yes_ask, bid_size, ask_size)
    micro_vs_mid = micro - mid
    depth_ratio = yes_depth / (no_depth + 1e-6)

    scores = []

    # Imbalance signal: strong buy pressure → YES
    if abs(imbalance) > 0.20:
        direction = +1 if imbalance > 0 else -1
        intensity = min(abs(imbalance), 0.80)
        scores.append(direction * intensity)

    # Microprice signal: if microprice > mid, smart money is buying YES
    if abs(micro_vs_mid) > 0.3:
        direction = +1 if micro_vs_mid > 0 else -1
        scores.append(direction * min(abs(micro_vs_mid) / 2.0, 0.50))

    # Depth ratio: more depth on YES side → price should rise
    if depth_ratio > 1.5:
        scores.append(+0.25)
    elif depth_ratio < 0.67:
        scores.append(-0.25)

    if not scores:
        return NONE_SIGNAL

    avg_score = float(np.mean(scores))
    if abs(avg_score) < 0.15:
        return NONE_SIGNAL

    signal = "YES" if avg_score > 0 else "NO"
    edge = min(abs(avg_score) * 0.18, 0.18)
    confidence = 0.51 + min(abs(avg_score) * 0.30, 0.30)

    logger.debug(
        f"[OrderBook] {market.get('ticker','?')} imbalance={imbalance:.2f} "
        f"micro_vs_mid={micro_vs_mid:.2f} depth_ratio={depth_ratio:.2f} → {signal}"
    )
    return {"signal": signal, "edge": round(edge, 4), "confidence": round(confidence, 4)}
