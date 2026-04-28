import logging
from kalshi_bot.client import KalshiClient, filter_tradeable

logger = logging.getLogger(__name__)


class MarketScanner:
    def __init__(self):
        self.client = KalshiClient()

    def scan(self) -> list[dict]:
        logger.info("Scanning active markets...")
        all_markets = self.client.get_all_active_markets()
        tradeable = filter_tradeable(all_markets)
        logger.info(f"Found {len(all_markets)} markets — {len(tradeable)} tradeable")
        return tradeable

    def enrich_with_history(self, market: dict) -> dict:
        """Add price momentum from recent trades (history endpoint 404s on all markets)."""
        trades = self.client.get_trades(market["ticker"], limit=50)
        market["history"] = []
        if len(trades) >= 2:
            prices = [
                round(float(t.get("yes_price_dollars", 0.5)) * 100)
                for t in sorted(trades, key=lambda t: t.get("created_time", ""))
            ]
            market["price_momentum"] = float(prices[-1] - prices[0])
        else:
            market["price_momentum"] = 0.0
        return market
