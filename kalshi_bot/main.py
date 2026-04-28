"""
Main orchestration loop for the Kalshi paper trading bot.
PAPER TRADING - NO REAL ORDERS
"""

import logging
import schedule
import time
from datetime import datetime, timezone

from kalshi_bot.config import (
    SCAN_INTERVAL_SECONDS,
    MIN_EDGE_THRESHOLD,
    MIN_CONFIDENCE,
)
from kalshi_bot.scanner import MarketScanner
from kalshi_bot.signals import mispricing, sentiment, ml_model
from kalshi_bot.signals import technical as tech_signal
from kalshi_bot.signals import orderbook as ob_signal
from kalshi_bot.risk import (
    position_size, can_trade, check_circuit_breaker, reset_daily_state,
)
from kalshi_bot.paper_trader import PaperTrader
from kalshi_bot import database as db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Signal weights (5 signals, sum = 1.0)
W_MISPRICING = 0.25
W_SENTIMENT  = 0.15
W_TECHNICAL  = 0.20
W_ORDERBOOK  = 0.10
W_ML         = 0.30

scanner = MarketScanner()
trader = PaperTrader()


ml_model.get_model()
logger.info("Stacking ensemble loaded/trained")



def combine_signals(sigs: dict) -> dict:
    weights = {
        "mispricing": W_MISPRICING,
        "sentiment":  W_SENTIMENT,
        "technical":  W_TECHNICAL,
        "orderbook":  W_ORDERBOOK,
        "ml":         W_ML,
    }

    def _score(s: dict) -> float:
        mul = {"YES": +1, "NO": -1}.get(s["signal"], 0)
        return mul * s["edge"] * s["confidence"]

    weighted_score = sum(weights[k] * _score(v) for k, v in sigs.items())

    # Only average edge/confidence over signals that actually fired (not NONE)
    active = {k: v for k, v in sigs.items() if v["signal"] != "NONE"}
    if active:
        total_w = sum(weights[k] for k in active)
        avg_edge = sum(weights[k] * active[k]["edge"] for k in active) / total_w
        avg_conf = sum(weights[k] * active[k]["confidence"] for k in active) / total_w
    else:
        avg_edge = 0.0
        avg_conf = 0.0

    return {
        "signal": "YES" if weighted_score > 0 else ("NO" if weighted_score < 0 else "NONE"),
        "edge": avg_edge,
        "confidence": avg_conf,
    }


def run_scan_cycle():
    logger.info("=" * 60)
    logger.info(f"Scan cycle — {datetime.now(timezone.utc).isoformat()}")

    # Circuit breaker check
    if check_circuit_breaker(trader.balance, trader.open_positions):
        logger.warning("Circuit breaker active — skipping this cycle")
        return

    markets = scanner.scan()
    if not markets:
        logger.info("No tradeable markets found")
        return

    active_tickers = {m["ticker"] for m in markets}
    settlements = scanner.client.get_settlements(max_pages=5)
    trader.check_expired_positions(active_tickers, settlements)

    trades_this_cycle = 0
    evaluated = 0

    # Sort by 24h volume descending — evaluate the most liquid markets first
    markets = sorted(markets, key=lambda m: m.get("volume_24h", 0), reverse=True)

    # Cap at 200 markets per cycle to avoid multi-minute scan loops
    MAX_EVAL_PER_CYCLE = 200

    for market in markets:
        ticker = market["ticker"]

        if market.get("category", "").strip() == "Sports":
            continue

        tradeable, reason = can_trade(
            ticker, trader.open_positions, trader.balance,
            market.get("category", "unknown")
        )
        if not tradeable:
            if reason not in ("already_in_market",):
                logger.debug(f"Skip {ticker}: {reason}")
            continue

        if evaluated >= MAX_EVAL_PER_CYCLE:
            break

        # Market already has price/volume/bid/ask from events API — no extra call needed
        market.setdefault("price_momentum", 0.0)
        market.setdefault("history", [])
        evaluated += 1

        # Run all 5 signals
        sig_m  = mispricing.analyze(market)
        sig_s  = sentiment.analyze(market)
        sig_t  = tech_signal.analyze(market)
        sig_ob = ob_signal.analyze(market)

        # Pass sentiment score to ML feature set
        s_compound = (sig_s["edge"] / 0.20) if sig_s["signal"] != "NONE" else 0.0
        if sig_s["signal"] == "NO":
            s_compound = -s_compound
        market["sentiment_score"] = s_compound

        sig_ml = ml_model.analyze(market)

        sigs = {
            "mispricing": sig_m,
            "sentiment":  sig_s,
            "technical":  sig_t,
            "orderbook":  sig_ob,
            "ml":         sig_ml,
        }

        combined = combine_signals(sigs)

        # Log all non-trivial signals
        for source, sig in sigs.items():
            if sig["edge"] > 0.01:
                db.log_signal(
                    ticker, sig["signal"], sig["edge"], sig["confidence"], source
                )

        if combined["signal"] == "NONE":
            continue
        if combined["edge"] < MIN_EDGE_THRESHOLD:
            continue
        if combined["confidence"] < MIN_CONFIDENCE:
            continue

        size = position_size(
            trader.balance,
            combined["confidence"],
            market["yes_price"],
            combined["signal"],
            trader.open_positions,
            market.get("category", "unknown"),
        )

        executed = trader.execute_paper_trade(market, combined["signal"], size)
        if executed:
            trades_this_cycle += 1

    summary = trader.get_portfolio_summary()
    logger.info(
        f"Cycle done | evaluated={evaluated} | trades={trades_this_cycle} | "
        f"balance=${summary['balance']:.2f} | P&L=${summary['total_pnl']:.2f} | "
        f"open={summary['open_positions']}"
    )


def main():
    logger.info("Kalshi Paper Trading Bot starting — PAPER TRADING - NO REAL ORDERS")
    reset_daily_state(trader.balance)
    schedule.every(SCAN_INTERVAL_SECONDS).seconds.do(run_scan_cycle)

    run_scan_cycle()

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
