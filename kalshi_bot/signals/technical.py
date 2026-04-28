"""
Technical analysis signals: RSI, Bollinger Bands, mean-reversion,
price velocity, and volatility regime detection.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

NONE_SIGNAL = {"signal": "NONE", "edge": 0.0, "confidence": 0.0}


# ── Indicator calculations ────────────────────────────────────────────────────

def _rsi(prices: list[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _bollinger(prices: list[float], period: int = 20, n_std: float = 2.0) -> dict:
    if len(prices) < period:
        mid = prices[-1] if prices else 50.0
        return {"mid": mid, "upper": mid + 10, "lower": mid - 10, "position": 0.5, "width": 0.2}
    window = prices[-period:]
    mid = float(np.mean(window))
    std = float(np.std(window))
    upper = mid + n_std * std
    lower = mid - n_std * std
    price = prices[-1]
    band_range = upper - lower
    position = (price - lower) / band_range if band_range > 0 else 0.5
    width = band_range / mid if mid > 0 else 0.2
    return {"mid": mid, "upper": upper, "lower": lower,
            "position": float(np.clip(position, 0, 1)), "width": width}


def _volatility(prices: list[float], period: int = 20) -> float:
    if len(prices) < 2:
        return 0.05
    log_returns = np.diff(np.log(np.clip(prices[-period:], 1, 99)))
    return float(np.std(log_returns)) if len(log_returns) > 0 else 0.05


def _momentum(prices: list[float], short: int = 5, long: int = 20) -> dict:
    if len(prices) < long:
        return {"fast": 0.0, "slow": 0.0, "velocity": 0.0}
    fast = prices[-1] - prices[-short] if len(prices) >= short else 0.0
    slow = prices[-1] - prices[-long]
    # Acceleration: rate of change of momentum
    velocity = (prices[-1] - prices[-2]) if len(prices) >= 2 else 0.0
    return {"fast": float(fast), "slow": float(slow), "velocity": float(velocity)}


def _mean_reversion_zscore(prices: list[float], period: int = 20) -> float:
    if len(prices) < period:
        return 0.0
    window = prices[-period:]
    mean = np.mean(window)
    std = np.std(window)
    if std < 0.5:
        return 0.0
    return float((prices[-1] - mean) / std)


# ── Feature extraction (for ML) ──────────────────────────────────────────────

def extract_features(market: dict) -> dict:
    """Extract all technical features. Returns dict of named features."""
    history = market.get("history", [])
    prices = [h["yes_price"] for h in history] if history else [market.get("yes_price", 50)]

    rsi = _rsi(prices)
    bb = _bollinger(prices)
    vol = _volatility(prices)
    mom = _momentum(prices)
    zscore = _mean_reversion_zscore(prices)

    return {
        "rsi": rsi,
        "bb_position": bb["position"],
        "bb_width": bb["width"],
        "price_vs_ma20": prices[-1] - bb["mid"],
        "price_vs_ma5": prices[-1] - float(np.mean(prices[-5:])) if len(prices) >= 5 else 0.0,
        "volatility": vol,
        "mom_fast": mom["fast"],
        "mom_slow": mom["slow"],
        "price_velocity": mom["velocity"],
        "mean_rev_zscore": zscore,
    }


# ── Signal generation ─────────────────────────────────────────────────────────

def analyze(market: dict) -> dict:
    history = market.get("history", [])
    if len(history) < 10:
        return NONE_SIGNAL

    prices = [h["yes_price"] for h in history]
    rsi = _rsi(prices)
    bb = _bollinger(prices)
    zscore = _mean_reversion_zscore(prices)
    mom = _momentum(prices)

    scores: list[tuple[float, float]] = []  # (directional score, weight)

    # RSI signal: oversold → YES, overbought → NO
    if rsi < 30:
        intensity = (30 - rsi) / 30.0
        scores.append((+intensity, 1.2))  # weighted more for strong oversold
    elif rsi > 70:
        intensity = (rsi - 70) / 30.0
        scores.append((-intensity, 1.2))
    else:
        scores.append((0.0, 0.3))

    # Bollinger position: below lower band → YES, above upper band → NO
    if bb["position"] < 0.10:
        scores.append((+(1.0 - bb["position"] * 10), 1.0))
    elif bb["position"] > 0.90:
        scores.append((-(bb["position"] - 0.90) * 10, 1.0))
    else:
        scores.append((0.0, 0.3))

    # Mean reversion z-score: mean-revert if |z| > 1.5
    if abs(zscore) > 1.5:
        direction = -1 if zscore > 0 else +1  # price too high → NO
        intensity = min(abs(zscore) / 3.0, 1.0)
        scores.append((direction * intensity, 1.0))
    else:
        scores.append((0.0, 0.4))

    # Momentum confirmation (short-term trend)
    if abs(mom["fast"]) > 3:
        direction = +1 if mom["fast"] > 0 else -1
        intensity = min(abs(mom["fast"]) / 15.0, 0.5)
        scores.append((direction * intensity, 0.5))

    # Weighted average score
    total_weight = sum(w for _, w in scores)
    if total_weight == 0:
        return NONE_SIGNAL
    weighted_score = sum(s * w for s, w in scores) / total_weight

    if abs(weighted_score) < 0.15:
        return NONE_SIGNAL

    signal = "YES" if weighted_score > 0 else "NO"
    edge = min(abs(weighted_score) * 0.25, 0.25)
    confidence = 0.50 + min(abs(weighted_score) * 0.35, 0.38)

    # Reduce confidence in high-volatility regimes
    vol = _volatility(prices)
    if vol > 0.15:
        confidence *= 0.90

    logger.debug(
        f"[Technical] {market.get('ticker','?')} rsi={rsi:.1f} bb={bb['position']:.2f} "
        f"z={zscore:.2f} → {signal} edge={edge:.3f}"
    )
    return {"signal": signal, "edge": round(edge, 4), "confidence": round(confidence, 4)}
