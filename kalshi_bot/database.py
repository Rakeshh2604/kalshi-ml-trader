import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = "kalshi_paper.db"


def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT NOT NULL,
                direction   TEXT NOT NULL,
                size        REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price  REAL,
                pnl         REAL,
                timestamp   TEXT NOT NULL,
                signal_source TEXT,
                status      TEXT DEFAULT 'open'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker     TEXT NOT NULL,
                signal     TEXT NOT NULL,
                edge       REAL,
                confidence REAL,
                source     TEXT,
                timestamp  TEXT NOT NULL
            )
        """)
        conn.commit()
    logger.info("Database initialized")


def log_trade(
    ticker: str,
    direction: str,
    size: float,
    entry_price: float,
    signal_source: str = "",
) -> int:
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO trades (ticker, direction, size, entry_price, timestamp, signal_source)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ticker, direction, size, entry_price, ts, signal_source),
        )
        conn.commit()
        return cur.lastrowid


def close_trade(trade_id: int, exit_price: float, pnl: float):
    with _connect() as conn:
        conn.execute(
            """UPDATE trades SET exit_price=?, pnl=?, status='closed' WHERE id=?""",
            (exit_price, pnl, trade_id),
        )
        conn.commit()


def log_signal(ticker: str, signal: str, edge: float, confidence: float, source: str):
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO signals (ticker, signal, edge, confidence, source, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ticker, signal, edge, confidence, source, ts),
        )
        conn.commit()


def get_trade_history(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_performance_stats() -> dict:
    with _connect() as conn:
        closed = conn.execute(
            "SELECT pnl FROM trades WHERE status='closed' AND pnl IS NOT NULL"
        ).fetchall()

    if not closed:
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "total_pnl": 0.0}

    pnls = [r["pnl"] for r in closed]
    wins = sum(1 for p in pnls if p > 0)
    return {
        "total_trades": len(pnls),
        "wins": wins,
        "losses": len(pnls) - wins,
        "win_rate": wins / len(pnls),
        "total_pnl": round(sum(pnls), 2),
    }
