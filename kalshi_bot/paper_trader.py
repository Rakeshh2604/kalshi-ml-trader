"""
Paper trading engine — simulates order execution with no real money.
PAPER TRADING - NO REAL ORDERS
"""

import logging
from datetime import datetime, timezone

from kalshi_bot import database as db
from kalshi_bot.config import STARTING_PAPER_BALANCE

logger = logging.getLogger(__name__)


class PaperTrader:
    def __init__(self):
        db.init_db()
        self.balance = STARTING_PAPER_BALANCE
        self.open_positions: dict[str, dict] = {}  # ticker -> position info
        self.trade_count = 0
        self.total_pnl = 0.0

    def execute_paper_trade(self, market: dict, signal: str, size: float) -> bool:
        """
        Simulate buying YES or NO contracts.
        PAPER TRADING - NO REAL ORDERS
        """
        ticker = market["ticker"]
        if ticker in self.open_positions:
            logger.debug(f"Already have position in {ticker}, skipping")
            return False

        if size > self.balance:
            logger.warning(f"Insufficient paper balance for {ticker}: need ${size:.2f}")
            return False

        yes_price = market["yes_price"]
        entry_price = yes_price if signal == "YES" else (100 - yes_price)

        # Deduct from balance
        self.balance -= size

        trade_id = db.log_trade(
            ticker=ticker,
            direction=signal,
            size=size,
            entry_price=entry_price,
            signal_source="combined",
        )

        self.open_positions[ticker] = {
            "trade_id": trade_id,
            "signal": signal,
            "size": size,
            "entry_price": entry_price,
            "close_time": market.get("close_time"),
        }

        self.trade_count += 1
        logger.info(
            f"[PAPER TRADE] {signal} {ticker} @ {entry_price}¢ | size=${size:.2f} | "
            f"balance=${self.balance:.2f}"
        )
        return True

    def close_position(self, ticker: str, resolved_yes: bool):
        """
        Close a position after market resolution.
        PAPER TRADING - NO REAL ORDERS
        """
        pos = self.open_positions.pop(ticker, None)
        if pos is None:
            return

        signal = pos["signal"]
        size = pos["size"]
        entry_price = pos["entry_price"]

        won = (signal == "YES" and resolved_yes) or (signal == "NO" and not resolved_yes)

        if won:
            # Contract pays $1 per contract; size was in dollars
            contracts = size / (entry_price / 100)
            payout = contracts * 1.0
            pnl = payout - size
            exit_price = 100
        else:
            pnl = -size
            exit_price = 0

        self.balance += size + pnl
        self.total_pnl += pnl

        db.close_trade(pos["trade_id"], exit_price, pnl)
        outcome = "WIN" if won else "LOSS"
        logger.info(
            f"[PAPER CLOSE] {outcome} {ticker} | pnl=${pnl:.2f} | balance=${self.balance:.2f}"
        )

    def check_expired_positions(self, active_tickers: set[str], settlements: list[dict] = None):
        """Close positions using real Kalshi settlement data where available."""
        now = datetime.now(timezone.utc)

        # Build a ticker→result map from real settlement data
        real_results: dict[str, bool] = {}
        for s in (settlements or []):
            ticker = s.get("market_ticker") or s.get("ticker", "")
            result = s.get("market_result") or s.get("result", "")
            if ticker and result in ("yes", "no"):
                real_results[ticker] = (result == "yes")

        expired = []
        for ticker, pos in self.open_positions.items():
            close_time = pos.get("close_time")
            if ticker in real_results:
                expired.append((ticker, real_results[ticker], "settled"))
            elif ticker not in active_tickers and (
                close_time and close_time < now
            ):
                expired.append((ticker, None, "expired_unknown"))

        for ticker, resolved_yes, reason in expired:
            if resolved_yes is not None:
                logger.info(f"[REAL SETTLEMENT] closing {ticker} → {'YES' if resolved_yes else 'NO'}")
                self.close_position(ticker, resolved_yes)
            else:
                # Market gone but no settlement data — mark as unknown, don't simulate
                pos = self.open_positions.pop(ticker, None)
                if pos:
                    db.close_trade(pos["trade_id"], exit_price=None, pnl=0.0)
                    logger.warning(f"[NO DATA] {ticker} expired with no settlement — marked neutral")

    def get_portfolio_summary(self) -> dict:
        stats = db.get_performance_stats()
        return {
            "balance": round(self.balance, 2),
            "starting_balance": STARTING_PAPER_BALANCE,
            "total_pnl": round(self.total_pnl, 2),
            "open_positions": len(self.open_positions),
            "open_tickers": list(self.open_positions.keys()),
            **stats,
        }
