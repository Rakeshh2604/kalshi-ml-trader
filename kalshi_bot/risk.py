"""
Risk management: Kelly criterion, portfolio heat, VaR-based sizing,
correlation-aware position caps, and daily loss circuit breaker.
"""

import logging
import numpy as np
from kalshi_bot.config import MAX_KELLY_FRACTION, MAX_POSITION_SIZE

logger = logging.getLogger(__name__)

# Portfolio-level risk limits
MAX_PORTFOLIO_HEAT = 0.60        # max fraction of balance deployed at once
MAX_OPEN_POSITIONS = 12          # hard cap on simultaneous positions
MAX_DAILY_LOSS_PCT = 0.20        # circuit breaker: halt if down >20% today
CORR_SAME_CATEGORY_PENALTY = 0.70  # reduce size by 30% for same-category positions

# Circuit breaker state
_daily_starting_balance: float | None = None
_circuit_broken: bool = False


def reset_daily_state(balance: float):
    global _daily_starting_balance, _circuit_broken
    _daily_starting_balance = balance
    _circuit_broken = False


def is_circuit_broken() -> bool:
    return _circuit_broken


def check_circuit_breaker(current_balance: float, open_positions: dict = None) -> bool:
    """Returns True if circuit breaker triggered. Uses total equity (cash + deployed), not just cash."""
    global _circuit_broken
    if _circuit_broken:
        return True
    if _daily_starting_balance is None:
        return False
    deployed = sum(p["size"] for p in (open_positions or {}).values())
    total_equity = current_balance + deployed
    daily_loss = (_daily_starting_balance - total_equity) / _daily_starting_balance
    if daily_loss >= MAX_DAILY_LOSS_PCT:
        _circuit_broken = True
        logger.warning(
            f"CIRCUIT BREAKER TRIGGERED: daily loss {daily_loss*100:.1f}% "
            f"exceeds {MAX_DAILY_LOSS_PCT*100:.0f}% limit. Halting new trades."
        )
        return True
    return False


def kelly_fraction(win_prob: float, odds: float) -> float:
    """f* = (b·p − q) / b"""
    q = 1.0 - win_prob
    if odds <= 0 or win_prob <= 0 or win_prob >= 1:
        return 0.0
    f = (odds * win_prob - q) / odds
    return max(f, 0.0)


def portfolio_heat(open_positions: dict, balance: float) -> float:
    """Fraction of (balance + deployed capital) currently in open positions."""
    deployed = sum(p["size"] for p in open_positions.values())
    total = balance + deployed
    return deployed / total if total > 0 else 0.0


def position_size(
    balance: float,
    win_prob: float,
    yes_price: int,
    signal: str,
    open_positions: dict | None = None,
    market_category: str = "unknown",
) -> float:
    """
    Return recommended dollar position size applying:
      1. Kelly criterion (capped at quarter-Kelly)
      2. Portfolio heat limit
      3. Same-category correlation penalty
      4. Hard cap at MAX_POSITION_SIZE
    """
    open_positions = open_positions or {}

    if signal == "YES":
        cost = yes_price / 100.0
        odds = (1.0 - cost) / cost if cost > 0 else 0.0
    elif signal == "NO":
        no_price = 100 - yes_price
        cost = no_price / 100.0
        odds = (1.0 - cost) / cost if cost > 0 else 0.0
    else:
        return 0.0

    f = kelly_fraction(win_prob, odds)
    f_capped = min(f, MAX_KELLY_FRACTION)

    # Portfolio heat limit: reduce size to stay below heat cap
    current_heat = portfolio_heat(open_positions, balance)
    available_heat = max(MAX_PORTFOLIO_HEAT - current_heat, 0.0)
    deployed = sum(p["size"] for p in open_positions.values())
    total_capital = balance + deployed
    heat_limited_size = available_heat * total_capital

    size = min(balance * f_capped, heat_limited_size)

    # Same-category correlation penalty
    open_cats = [p.get("category", "unknown") for p in open_positions.values()]
    same_cat_count = open_cats.count(market_category.lower())
    if same_cat_count > 0:
        penalty = CORR_SAME_CATEGORY_PENALTY ** same_cat_count
        size *= penalty
        logger.debug(
            f"Correlation penalty ×{penalty:.2f} for {same_cat_count} "
            f"existing {market_category} positions"
        )

    size = min(size, MAX_POSITION_SIZE)
    if size < 5.0:  # skip trivially small positions — not worth a slot
        return 0.0
    return round(size, 2)


def can_trade(
    ticker: str,
    open_positions: dict,
    balance: float,
    market_category: str = "unknown",
) -> tuple[bool, str]:
    """
    Returns (can_trade_bool, reason_string).
    """
    if ticker in open_positions:
        return False, "already_in_market"
    if _circuit_broken:
        return False, "circuit_breaker"
    if len(open_positions) >= MAX_OPEN_POSITIONS:
        return False, f"max_positions_{MAX_OPEN_POSITIONS}"
    heat = portfolio_heat(open_positions, balance)
    if heat >= MAX_PORTFOLIO_HEAT:
        return False, f"portfolio_heat_{heat:.0%}"
    return True, "ok"


def var_estimate(trade_pnls: list[float], confidence: float = 0.95) -> float:
    """Historical VaR: worst loss at confidence level (positive = loss)."""
    if not trade_pnls:
        return 0.0
    arr = np.array(trade_pnls)
    return float(-np.percentile(arr, (1 - confidence) * 100))
