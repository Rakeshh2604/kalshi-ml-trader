"""
Real-data collection pipeline.

Runs continuously and:
  1. Every SNAPSHOT_INTERVAL_SECONDS: snapshots all active markets
     (price, volume, orderbook, spread) into SQLite
  2. Every HISTORY_INTERVAL_SECONDS: fetches full price history for
     each active market and stores it
  3. Every RESOLUTION_INTERVAL_SECONDS: checks for newly settled markets,
     labels them YES/NO, and saves to the training dataset
  4. Exports a clean training CSV whenever N new resolved markets are collected

Usage:
    python -m kalshi_bot.data_collector          # run continuously
    python -m kalshi_bot.data_collector --export # export training CSV and exit
    python -m kalshi_bot.data_collector --status # print collection stats
"""

import sys
import time
import json
import logging
import sqlite3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from kalshi_bot.client import KalshiClient
from kalshi_bot.signals.technical import extract_features as tech_features
from kalshi_bot.signals.orderbook import extract_features as ob_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH            = "kalshi_bot/market_data.db"
TRAINING_CSV_PATH  = "kalshi_bot/training_data.csv"
SNAPSHOT_INTERVAL  = 120     # seconds between market snapshots
HISTORY_INTERVAL   = 600     # seconds between full history pulls
RESOLUTION_INTERVAL = 300    # seconds between settlement checks
EXPORT_EVERY_N     = 50      # auto-export after this many new resolved markets


# ── Database setup ────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT NOT NULL,
                title           TEXT,
                category        TEXT,
                yes_price       INTEGER,
                yes_bid         INTEGER,
                yes_ask         INTEGER,
                spread          INTEGER,
                volume          INTEGER,
                volume_24h      INTEGER,
                open_interest   INTEGER,
                liquidity       INTEGER,
                hours_to_close  REAL,
                close_time      TEXT,
                ob_bid_depth    REAL,
                ob_ask_depth    REAL,
                ob_imbalance    REAL,
                microprice      REAL,
                captured_at     TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT NOT NULL,
                yes_price   INTEGER,
                trade_ts    TEXT,
                captured_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resolved_markets (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT UNIQUE NOT NULL,
                title           TEXT,
                category        TEXT,
                result          TEXT,          -- 'yes' or 'no'
                resolved_at     TEXT,
                -- snapshot features at collection time
                yes_price_avg   REAL,
                yes_price_last  REAL,
                volume_total    INTEGER,
                spread_avg      REAL,
                hours_to_close  REAL,
                ob_imbalance_avg REAL,
                price_momentum  REAL,
                price_volatility REAL,
                n_snapshots     INTEGER,
                raw_json        TEXT           -- full feature dict
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_snap_ticker ON market_snapshots(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_ticker ON price_history(ticker)")
        conn.commit()
    logger.info("Data collection DB initialized")


# ── Snapshot logic ────────────────────────────────────────────────────────────

def _ob_depth(levels: list) -> float:
    return sum(
        level[1] if isinstance(level, (list, tuple)) and len(level) > 1 else 0
        for level in levels[:5]
    )


def snapshot_markets(client: KalshiClient, ob_limit: int = 30) -> int:
    """
    Snapshot all active markets. Orderbook is only fetched for the
    top ob_limit markets by volume to avoid rate limits with 10k+ markets.
    """
    markets = client.get_all_active_markets()
    if not markets:
        return 0

    now = datetime.now(timezone.utc).isoformat()

    # Sort by volume desc; only fetch orderbooks for top markets
    markets_sorted = sorted(markets, key=lambda m: m["volume"], reverse=True)
    ob_tickers = {m["ticker"] for m in markets_sorted[:ob_limit]}

    rows = []
    for m in markets:
        ticker = m["ticker"]
        bid_depth = ask_depth = imbalance = microprice = 0.0

        if ticker in ob_tickers:
            ob = client.get_orderbook(ticker, depth=5)
            if ob:
                bid_depth = _ob_depth(ob.get("yes", []))
                ask_depth = _ob_depth(ob.get("no", []))
                total = bid_depth + ask_depth
                imbalance = (bid_depth - ask_depth) / total if total > 0 else 0.0
                yb = m["yes_bid"]
                ya = m["yes_ask"]
                microprice = (ya * bid_depth + yb * ask_depth) / total if total > 0 else (yb + ya) / 2

        rows.append((
            ticker, m["title"], m["category"],
            m["yes_price"], m["yes_bid"], m["yes_ask"],
            m["yes_ask"] - m["yes_bid"],
            m["volume"], m["volume_24h"], m["open_interest"], m["liquidity"],
            m["hours_to_close"],
            m["close_time"].isoformat() if m["close_time"] else None,
            bid_depth, ask_depth, imbalance, microprice,
            now,
        ))

    with _connect() as conn:
        conn.executemany("""
            INSERT INTO market_snapshots
            (ticker, title, category, yes_price, yes_bid, yes_ask, spread,
             volume, volume_24h, open_interest, liquidity, hours_to_close,
             close_time, ob_bid_depth, ob_ask_depth, ob_imbalance, microprice,
             captured_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
        conn.commit()

    logger.info(f"Snapshotted {len(rows)} markets (OB fetched for top {ob_limit})")
    return len(rows)


# ── History collection ────────────────────────────────────────────────────────

def collect_history(client: KalshiClient, markets: list[dict]) -> int:
    now = datetime.now(timezone.utc).isoformat()
    total = 0

    for m in markets:
        ticker = m["ticker"]
        history = client.get_market_history(ticker, limit=200)
        if not history:
            continue

        rows = [
            (ticker, h["yes_price"], str(h.get("timestamp", "")), now)
            for h in history
        ]

        with _connect() as conn:
            # Avoid duplicates by checking existing count
            existing = conn.execute(
                "SELECT COUNT(*) FROM price_history WHERE ticker=?", (ticker,)
            ).fetchone()[0]

            # Only insert new entries (by timestamp)
            new_rows = []
            existing_ts = set(
                r[0] for r in conn.execute(
                    "SELECT trade_ts FROM price_history WHERE ticker=?", (ticker,)
                ).fetchall()
            )
            for row in rows:
                if str(row[2]) not in existing_ts:
                    new_rows.append(row)

            if new_rows:
                conn.executemany(
                    "INSERT INTO price_history (ticker, yes_price, trade_ts, captured_at) VALUES (?,?,?,?)",
                    new_rows,
                )
                conn.commit()
                total += len(new_rows)
                logger.debug(f"  {ticker}: +{len(new_rows)} history bars")

    logger.info(f"History collection: {total} new bars across {len(markets)} markets")
    return total


# ── Resolution checker ────────────────────────────────────────────────────────

def _build_feature_row(ticker: str) -> dict:
    """Aggregate snapshot history for a ticker into ML feature row."""
    with _connect() as conn:
        snaps = conn.execute(
            """SELECT yes_price, volume, spread, ob_imbalance, hours_to_close
               FROM market_snapshots WHERE ticker=? ORDER BY captured_at""",
            (ticker,)
        ).fetchall()

        hist = conn.execute(
            "SELECT yes_price FROM price_history WHERE ticker=? ORDER BY trade_ts",
            (ticker,)
        ).fetchall()

    if not snaps:
        return {}

    prices     = [s["yes_price"] for s in snaps]
    volumes    = [s["volume"] for s in snaps]
    spreads    = [s["spread"] for s in snaps]
    imbalances = [s["ob_imbalance"] for s in snaps]
    htc_vals   = [s["hours_to_close"] for s in snaps]

    hist_prices = [h["yes_price"] for h in hist] if hist else prices

    # Momentum: last price minus first price in history
    momentum = hist_prices[-1] - hist_prices[0] if len(hist_prices) >= 2 else 0.0

    # Volatility: std of log-returns in price history
    if len(hist_prices) >= 5:
        log_r = np.diff(np.log(np.clip(hist_prices, 1, 99)))
        volatility = float(np.std(log_r))
    else:
        volatility = 0.05

    return {
        "yes_price_avg":    float(np.mean(prices)),
        "yes_price_last":   float(prices[-1]),
        "volume_total":     int(max(volumes)),
        "spread_avg":       float(np.mean(spreads)),
        "hours_to_close":   float(htc_vals[-1]) if htc_vals else 0.0,
        "ob_imbalance_avg": float(np.mean(imbalances)),
        "price_momentum":   momentum,
        "price_volatility": volatility,
        "n_snapshots":      len(snaps),
    }


def check_resolutions(client: KalshiClient) -> int:
    """Poll for recently settled markets and record them."""
    settlements = client.get_settlements(max_pages=20)
    if not settlements:
        return 0

    new_resolutions = 0

    with _connect() as conn:
        already_done = set(
            r[0] for r in conn.execute(
                "SELECT ticker FROM resolved_markets"
            ).fetchall()
        )

    for s in settlements:
        ticker = s.get("market_ticker") or s.get("ticker", "")
        if not ticker or ticker in already_done:
            continue

        result = (
            s.get("market_result")
            or s.get("yes_outcome")
            or s.get("result")
        )
        if result not in ("yes", "no"):
            continue

        result = "yes" if result in ("yes", True) else "no"
        resolved_at = s.get("settled_time") or datetime.now(timezone.utc).isoformat()

        # Get category from our snapshot first (faster, no API call needed)
        with _connect() as _c:
            snap_row = _c.execute(
                "SELECT title, category FROM market_snapshots WHERE ticker=? LIMIT 1", (ticker,)
            ).fetchone()
        title    = snap_row["title"] if snap_row else s.get("ticker", ticker)
        category = snap_row["category"] if snap_row and snap_row["category"] != "unknown" else "unknown"

        # Fall back to API only if we don't have it
        market = None
        if category == "unknown":
            market = client.get_market(ticker)
            if market:
                title    = market["title"] or title
                category = market["category"] or "unknown"

        # Build features: prefer snapshot history, fall back to fresh market data
        # Pull features from snapshots we collected while market was active
        feats = _build_feature_row(ticker)

        with _connect() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO resolved_markets
                (ticker, title, category, result, resolved_at,
                 yes_price_avg, yes_price_last, volume_total, spread_avg,
                 hours_to_close, ob_imbalance_avg, price_momentum,
                 price_volatility, n_snapshots, raw_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ticker, title, category, result, str(resolved_at),
                feats.get("yes_price_avg"),
                feats.get("yes_price_last"),
                feats.get("volume_total"),
                feats.get("spread_avg"),
                feats.get("hours_to_close"),
                feats.get("ob_imbalance_avg"),
                feats.get("price_momentum"),
                feats.get("price_volatility"),
                feats.get("n_snapshots"),
                json.dumps(feats),
            ))
            conn.commit()

        new_resolutions += 1
        outcome_str = "✓ YES" if result == "yes" else "✗ NO"
        logger.info(f"Resolved: {ticker} → {outcome_str} | {title[:40]}")

    if new_resolutions:
        logger.info(f"Recorded {new_resolutions} new resolutions")
    return new_resolutions


# ── Training data export ──────────────────────────────────────────────────────

def export_training_data(min_snapshots: int = 3) -> pd.DataFrame:
    """
    Build ML training DataFrame from resolved markets.
    Only includes markets with at least min_snapshots data points.
    """
    with _connect() as conn:
        rows = conn.execute("""
            SELECT ticker, title, category, result,
                   yes_price_avg, yes_price_last, volume_total, spread_avg,
                   hours_to_close, ob_imbalance_avg, price_momentum,
                   price_volatility, n_snapshots
            FROM resolved_markets
            WHERE n_snapshots >= ?
        """, (min_snapshots,)).fetchall()

    if not rows:
        logger.warning("No resolved markets with enough snapshots yet")
        return pd.DataFrame()

    records = []
    for r in rows:
        yp = r["yes_price_avg"] or 50.0
        implied_prob = yp / 100.0
        import math
        logit = math.log(implied_prob / (1 - implied_prob + 1e-9) + 1e-9)

        # Encode category
        CATEGORIES = [
            "politics", "economics", "sports", "crypto", "weather",
            "entertainment", "science", "finance", "unknown",
        ]
        cat = (r["category"] or "unknown").lower()
        cat_idx = CATEGORIES.index(cat) if cat in CATEGORIES else CATEGORIES.index("unknown")

        records.append({
            "ticker":            r["ticker"],
            "yes_price":         yp,
            "implied_prob":      implied_prob,
            "logit_prob":        logit,
            "log_volume":        math.log1p(r["volume_total"] or 0),
            "spread":            r["spread_avg"] or 2.0,
            "relative_spread":   (r["spread_avg"] or 2.0) / (yp + 1e-6),
            "log_hours_to_close": math.log1p(r["hours_to_close"] or 1),
            "category":          float(cat_idx),
            "sentiment_score":   0.0,        # not available at resolution time
            "price_momentum":    r["price_momentum"] or 0.0,
            "mom_fast":          0.0,
            "rsi":               50.0,        # not easily reconstructed
            "bb_position":       0.5,
            "bb_width":          0.2,
            "price_vs_ma20":     (r["yes_price_last"] or yp) - yp,
            "price_vs_ma5":      (r["yes_price_last"] or yp) - yp,
            "volatility":        r["price_volatility"] or 0.05,
            "mean_reversion_z":  0.0,
            "ob_imbalance":      r["ob_imbalance_avg"] or 0.0,
            "microprice_vs_mid": 0.0,
            "depth_ratio":       0.0,
            "spread_vol_interaction": (r["spread_avg"] or 2.0) * math.log1p(r["volume_total"] or 0) / 20.0,
            "sentiment_x_prob":  0.0,
            "rsi_x_bb":          25.0,
            "rev_x_vol":         0.0,
            "label":             1 if r["result"] == "yes" else 0,
        })

    df = pd.DataFrame(records)
    df.to_csv(TRAINING_CSV_PATH, index=False)
    logger.info(f"Exported {len(df)} training samples to {TRAINING_CSV_PATH}")
    return df


# ── Stats printer ─────────────────────────────────────────────────────────────

def print_status():
    with _connect() as conn:
        n_snap = conn.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0]
        n_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM market_snapshots").fetchone()[0]
        n_hist = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
        n_resolved = conn.execute("SELECT COUNT(*) FROM resolved_markets").fetchone()[0]
        n_yes = conn.execute("SELECT COUNT(*) FROM resolved_markets WHERE result='yes'").fetchone()[0]
        n_no  = conn.execute("SELECT COUNT(*) FROM resolved_markets WHERE result='no'").fetchone()[0]
        last_snap = conn.execute(
            "SELECT captured_at FROM market_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        categories = conn.execute(
            "SELECT category, COUNT(*) as n FROM resolved_markets GROUP BY category ORDER BY n DESC"
        ).fetchall()

    print("\n" + "=" * 56)
    print("  KALSHI DATA COLLECTOR STATUS")
    print("=" * 56)
    print(f"  Snapshots collected   : {n_snap:>8,}  ({n_tickers} unique tickers)")
    print(f"  Price history bars    : {n_hist:>8,}")
    print(f"  Resolved markets      : {n_resolved:>8,}  ({n_yes} YES / {n_no} NO)")
    print(f"  Last snapshot         : {last_snap[0] if last_snap else 'never'}")
    if categories:
        print(f"\n  Resolved by category:")
        for row in categories:
            print(f"    {row[0]:<16} {row[1]:>4}")
    print(f"\n  Training data ready at: {TRAINING_CSV_PATH}")
    print(f"  {'Ready to retrain ML!' if n_resolved >= 100 else f'Need {100 - n_resolved} more resolved markets for reliable ML retraining'}")
    print("=" * 56 + "\n")


# ── Main collection loop ──────────────────────────────────────────────────────

def run_collector():
    init_db()
    client = KalshiClient()

    if not client._is_online():
        logger.error("Cannot start collector: no API credentials configured")
        return

    last_snapshot   = 0.0
    last_history    = 0.0
    last_resolution = 0.0
    total_resolved  = 0
    last_export_count = 0

    logger.info("Data collector started — collecting real Kalshi market data")
    print_status()

    while True:
        now = time.time()

        # ── Market snapshots
        if now - last_snapshot >= SNAPSHOT_INTERVAL:
            n = snapshot_markets(client)
            last_snapshot = now

        # History endpoint returns 404 for all markets — skipped.
        last_history = now

        # ── Resolution check
        if now - last_resolution >= RESOLUTION_INTERVAL:
            n_new = check_resolutions(client)
            total_resolved += n_new
            last_resolution = now

            # Auto-export every EXPORT_EVERY_N new resolutions
            with _connect() as conn:
                current_count = conn.execute(
                    "SELECT COUNT(*) FROM resolved_markets"
                ).fetchone()[0]
            if current_count - last_export_count >= EXPORT_EVERY_N:
                export_training_data()
                last_export_count = current_count
                logger.info(f"Auto-exported training data ({current_count} resolved markets)")

        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi real data collector")
    parser.add_argument("--export", action="store_true", help="Export training CSV and exit")
    parser.add_argument("--status", action="store_true", help="Print collection status and exit")
    parser.add_argument("--snapshot", action="store_true", help="Take one snapshot and exit")
    args = parser.parse_args()

    init_db()
    client = KalshiClient()

    if args.status:
        print_status()
    elif args.export:
        df = export_training_data()
        print(f"\nExported {len(df)} samples to {TRAINING_CSV_PATH}")
    elif args.snapshot:
        n = snapshot_markets(client)
        markets = client.get_all_active_markets()
        collect_history(client, markets[:50])
        check_resolutions(client)
        print_status()
    else:
        run_collector()
