"""
Historical data backfill using Kalshi's tick-by-tick trades endpoint.

Kalshi is CFTC-regulated and publishes full trade history per market.
This script:
  1. Fetches all trades for every resolved market (up to 5 pages / ~1000 trades each)
  2. Reconstructs OHLCV + technical features from real price trajectories
  3. Builds a clean training dataset with real labels

Usage:
    python -m kalshi_bot.historical_backfill
"""

import time
import sqlite3
import logging
import math
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from kalshi_bot.client import KalshiClient, _request, _load_auth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "kalshi_bot/market_data.db"
TRAINING_CSV = "kalshi_bot/training_data_real.csv"
MAX_TRADE_PAGES = 5   # up to 1000 trades per market
RATE_LIMIT_SLEEP = 0.15  # seconds between API calls

CATEGORIES = [
    "Politics", "Elections", "Sports", "Crypto", "Climate and Weather",
    "Entertainment", "Economics", "Science and Technology", "Financials",
    "Companies", "Commodities", "Mentions", "Health", "Social", "World", "unknown",
]


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_all_trades(auth, ticker: str) -> list[dict]:
    """Fetch up to MAX_TRADE_PAGES * 200 trades for a market."""
    trades = []
    cursor = None
    for _ in range(MAX_TRADE_PAGES):
        params = {"ticker": ticker, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = _request(auth, "GET", "/markets/trades", params=params)
        if not data:
            break
        batch = data.get("trades") or []
        trades.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
        time.sleep(RATE_LIMIT_SLEEP)
    return trades


def _trades_to_features(trades: list[dict], category: str, yes_price_snapshot: float) -> dict:
    """Convert raw trade list into ML features."""
    if not trades:
        return {}

    # Sort oldest → newest
    trades_sorted = sorted(trades, key=lambda t: t.get("created_time", ""))

    prices = [round(float(t.get("yes_price_dollars", 0.5)) * 100) for t in trades_sorted]
    sizes  = [float(t.get("count_fp", 1)) for t in trades_sorted]
    sides  = [t.get("taker_side", "yes") for t in trades_sorted]

    prices_arr = np.array(prices, dtype=float)
    sizes_arr  = np.array(sizes, dtype=float)

    # ── Price features ────────────────────────────────────────────
    yes_price_first = float(prices_arr[0])
    yes_price_last  = float(prices_arr[-1])
    yes_price_avg   = float(np.average(prices_arr, weights=sizes_arr))
    yes_price_high  = float(prices_arr.max())
    yes_price_low   = float(prices_arr.min())
    price_range     = yes_price_high - yes_price_low

    # Momentum: last vs first price
    price_momentum = yes_price_last - yes_price_first

    # Short-term momentum (last 20% of trades)
    n = len(prices)
    recent_cutoff = max(1, int(n * 0.8))
    mom_fast = float(prices_arr[-1] - prices_arr[recent_cutoff])

    # ── Volume features ───────────────────────────────────────────
    total_volume = float(sizes_arr.sum())
    yes_volume = sum(s for s, side in zip(sizes, sides) if side == "yes")
    no_volume  = sum(s for s, side in zip(sizes, sides) if side == "no")
    volume_imbalance = (yes_volume - no_volume) / (total_volume + 1e-6)

    # ── Volatility ────────────────────────────────────────────────
    if len(prices) >= 5:
        log_returns = np.diff(np.log(np.clip(prices_arr, 1, 99)))
        volatility = float(np.std(log_returns))
    else:
        volatility = 0.05

    # ── RSI (14-period on trade prices) ──────────────────────────
    def _rsi(p, period=14):
        if len(p) < period + 1:
            return 50.0
        deltas = np.diff(p)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        ag = np.mean(gains[:period])
        al = np.mean(losses[:period])
        for g, l in zip(gains[period:], losses[period:]):
            ag = (ag * (period - 1) + g) / period
            al = (al * (period - 1) + l) / period
        return 100 - (100 / (1 + ag / al)) if al > 0 else 100.0

    rsi = _rsi(prices_arr)

    # ── Bollinger position ────────────────────────────────────────
    period = min(20, len(prices))
    window = prices_arr[-period:]
    bb_mid = float(np.mean(window))
    bb_std = float(np.std(window))
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = bb_upper - bb_lower
    bb_position = (yes_price_last - bb_lower) / bb_range if bb_range > 0 else 0.5
    bb_width    = bb_range / bb_mid if bb_mid > 0 else 0.2

    # ── Mean reversion z-score ────────────────────────────────────
    mean_rev_z = (yes_price_last - bb_mid) / bb_std if bb_std > 0 else 0.0

    # ── Spread proxy (from price range over last 20 trades) ───────
    recent_prices = prices_arr[-20:] if len(prices) >= 20 else prices_arr
    spread_est = float(recent_prices.max() - recent_prices.min())
    spread_est = max(1.0, min(spread_est, 20.0))

    # ── Category encoding ─────────────────────────────────────────
    cat_norm = category.strip()
    cat_idx  = CATEGORIES.index(cat_norm) if cat_norm in CATEGORIES else CATEGORIES.index("unknown")

    # ── Derived features ──────────────────────────────────────────
    implied_prob = yes_price_avg / 100.0
    logit_prob   = math.log(implied_prob / (1 - implied_prob + 1e-9) + 1e-9)
    log_volume   = math.log1p(total_volume)
    rel_spread   = spread_est / (yes_price_avg + 1e-6)

    return {
        "n_trades":          n,
        "yes_price":         yes_price_avg,
        "implied_prob":      implied_prob,
        "logit_prob":        logit_prob,
        "log_volume":        log_volume,
        "spread":            spread_est,
        "relative_spread":   rel_spread,
        "log_hours_to_close": 0.0,          # unknown at resolution
        "category":          float(cat_idx),
        "sentiment_score":   0.0,
        "price_momentum":    price_momentum,
        "mom_fast":          mom_fast,
        "rsi":               rsi,
        "bb_position":       float(np.clip(bb_position, 0, 1)),
        "bb_width":          bb_width,
        "price_vs_ma20":     yes_price_last - bb_mid,
        "price_vs_ma5":      yes_price_last - float(np.mean(prices_arr[-5:])) if n >= 5 else 0.0,
        "volatility":        volatility,
        "mean_reversion_z":  mean_rev_z,
        "ob_imbalance":      volume_imbalance,
        "microprice_vs_mid": yes_price_last - yes_price_avg,
        "depth_ratio":       math.log1p(yes_volume / (no_volume + 1e-6)),
        "spread_vol_interaction": spread_est * log_volume / 20.0,
        "sentiment_x_prob":  0.0,
        "rsi_x_bb":          rsi * float(np.clip(bb_position, 0, 1)),
        "rev_x_vol":         abs(mean_rev_z) * volatility,
        # Extra real-data features
        "price_first":       yes_price_first,
        "price_last":        yes_price_last,
        "price_high":        yes_price_high,
        "price_low":         yes_price_low,
        "price_range":       price_range,
        "volume_imbalance":  volume_imbalance,
        "total_volume":      total_volume,
    }


def run_backfill(max_markets: int = 2000) -> pd.DataFrame:
    auth = _load_auth()
    if not auth:
        logger.error("No API credentials")
        return pd.DataFrame()

    # Get all resolved markets from DB
    with _connect() as conn:
        resolved = conn.execute(
            "SELECT ticker, title, category, result FROM resolved_markets LIMIT ?",
            (max_markets,)
        ).fetchall()

    logger.info(f"Backfilling trade history for {len(resolved)} resolved markets...")

    records = []
    errors  = 0

    for i, row in enumerate(resolved):
        ticker   = row["ticker"]
        result   = row["result"]
        category = row["category"] or "unknown"

        trades = _fetch_all_trades(auth, ticker)

        if not trades:
            errors += 1
            logger.debug(f"  No trades: {ticker}")
            continue

        feats = _trades_to_features(trades, category, yes_price_snapshot=50.0)
        if not feats:
            continue

        feats["ticker"] = ticker
        feats["title"]  = row["title"] or ""
        feats["label"]  = 1 if result == "yes" else 0

        records.append(feats)

        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(resolved)} processed ({len(records)} with data, {errors} empty)")

        time.sleep(RATE_LIMIT_SLEEP)

    if not records:
        logger.error("No records built — all trades returned empty")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Drop ticker/title before saving ML features
    meta_cols = ["ticker", "title", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    logger.info(f"\nDataset built: {len(df)} markets, {len(feature_cols)} features")
    logger.info(f"Label balance: YES={df.label.sum()} NO={len(df)-df.label.sum()}")

    # Save
    df.to_csv(TRAINING_CSV, index=False)
    logger.info(f"Saved to {TRAINING_CSV}")

    # Print summary stats
    print(f"\n{'='*60}")
    print(f"  REAL HISTORICAL TRAINING DATA SUMMARY")
    print(f"{'='*60}")
    print(f"  Markets with trade data  : {len(df):>6}")
    print(f"  Markets with no trades   : {errors:>6}")
    print(f"  Label balance            : YES={df.label.sum()} / NO={len(df)-df.label.sum()}")
    print(f"  Avg trades per market    : {df['n_trades'].mean():>6.0f}")
    print(f"  Avg total volume         : {df['total_volume'].mean():>6.0f} contracts")
    print(f"  Avg price momentum       : {df['price_momentum'].mean():>+6.2f}¢")
    print(f"  Avg volatility           : {df['volatility'].mean():>6.4f}")
    print()
    print(f"  Feature correlations with label (top 10):")
    num_df = df[feature_cols + ["label"]].select_dtypes(include=[float, int])
    corrs = num_df.corr()["label"].drop("label").abs().sort_values(ascending=False)
    for feat, corr in corrs.head(10).items():
        print(f"    {feat:<30} {corr:>6.4f}")
    print(f"{'='*60}\n")

    return df


if __name__ == "__main__":
    run_backfill()
