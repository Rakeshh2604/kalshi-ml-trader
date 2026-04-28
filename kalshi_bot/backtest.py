"""
Backtesting engine using synthetic market data.

Simulates 500 markets, runs all three signals, applies risk management,
and reports realistic performance metrics.

Usage:
    python -m kalshi_bot.backtest
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from kalshi_bot.signals import mispricing, ml_model
from kalshi_bot.risk import kelly_fraction, position_size
from kalshi_bot.config import (
    STARTING_PAPER_BALANCE,
    MIN_EDGE_THRESHOLD,
    MIN_CONFIDENCE,
    MAX_POSITION_SIZE,
    WEIGHT_MISPRICING,
    WEIGHT_SENTIMENT,
    WEIGHT_ML,
)

# Make ML model available
_ml = ml_model.get_model()

CATEGORIES = [
    "politics", "economics", "sports", "crypto",
    "weather", "entertainment", "finance",
]

rng = np.random.default_rng(2024)


# ── Synthetic market generator ────────────────────────────────────────────────

def _generate_price_history(true_prob: float, n: int = 30, noise: float = 8.0) -> list[dict]:
    """Simulate price history that mean-reverts around true_prob."""
    prices = []
    price = rng.integers(10, 91)
    for _ in range(n):
        drift = (true_prob * 100 - price) * 0.05
        price = int(np.clip(price + drift + rng.normal(0, noise), 5, 95))
        prices.append({"yes_price": price, "timestamp": None})
    return prices


def generate_market(idx: int) -> dict:
    true_prob = float(rng.uniform(0.10, 0.90))
    yes_price = int(np.clip(true_prob * 100 + rng.normal(0, 6), 5, 95))
    spread = int(rng.integers(1, 12))
    category = random.choice(CATEGORIES)
    hours_to_close = float(rng.uniform(2, 200))
    volume = int(rng.integers(1500, 60_000))
    momentum = float(rng.normal(0, 5))
    sentiment_score = float(rng.uniform(-0.8, 0.8))

    history = _generate_price_history(true_prob, n=25)
    # Inject a clear mispricing in ~30 % of markets
    if rng.random() < 0.30:
        direction = 1 if true_prob > yes_price / 100 else -1
        for h in history[-8:]:
            h["yes_price"] = int(np.clip(h["yes_price"] - direction * rng.integers(8, 18), 5, 95))

    return {
        "ticker": f"SIM-{idx:04d}",
        "title": f"Simulated market {idx}",
        "yes_price": yes_price,
        "yes_bid": yes_price - spread // 2,
        "yes_ask": yes_price + spread // 2,
        "no_price": 100 - yes_price,
        "volume": volume,
        "hours_to_close": hours_to_close,
        "category": category,
        "price_momentum": momentum,
        "sentiment_score": sentiment_score,
        "history": history,
        "true_prob": true_prob,          # ground truth (not available to signals)
        "resolves_yes": rng.random() < true_prob,
    }


# ── Signal helpers ────────────────────────────────────────────────────────────

def _stub_sentiment(market: dict) -> dict:
    """Stub sentiment signal from pre-computed sentiment_score."""
    score = market.get("sentiment_score", 0.0)
    if abs(score) < 0.10:
        return {"signal": "NONE", "edge": 0.0, "confidence": 0.0}
    edge = min(abs(score) * 0.20, 0.20)
    confidence = 0.50 + min(abs(score) * 0.25, 0.35)
    signal = "YES" if score > 0 else "NO"
    return {"signal": signal, "edge": edge, "confidence": confidence}


def combine(sig_m: dict, sig_s: dict, sig_ml: dict) -> dict:
    def _score(s: dict) -> float:
        mul = 1 if s["signal"] == "YES" else (-1 if s["signal"] == "NO" else 0)
        return mul * s["edge"] * s["confidence"]

    score = (
        WEIGHT_MISPRICING * _score(sig_m)
        + WEIGHT_SENTIMENT * _score(sig_s)
        + WEIGHT_ML * _score(sig_ml)
    )
    avg_edge = (
        WEIGHT_MISPRICING * sig_m["edge"]
        + WEIGHT_SENTIMENT * sig_s["edge"]
        + WEIGHT_ML * sig_ml["edge"]
    )
    avg_conf = (
        WEIGHT_MISPRICING * sig_m["confidence"]
        + WEIGHT_SENTIMENT * sig_s["confidence"]
        + WEIGHT_ML * sig_ml["confidence"]
    )

    return {
        "signal": "YES" if score > 0 else ("NO" if score < 0 else "NONE"),
        "edge": avg_edge,
        "confidence": avg_conf,
    }


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_backtest(n_markets: int = 500, starting_balance: float = STARTING_PAPER_BALANCE) -> dict:
    balance = starting_balance
    trades = []
    signals_fired = 0

    for i in range(n_markets):
        market = generate_market(i)

        sig_m = mispricing.analyze(market)
        sig_s = _stub_sentiment(market)
        sig_ml = _ml.predict(market)

        combined = combine(sig_m, sig_s, sig_ml)

        if combined["signal"] == "NONE":
            continue
        if combined["edge"] < MIN_EDGE_THRESHOLD:
            continue
        if combined["confidence"] < MIN_CONFIDENCE:
            continue

        signals_fired += 1

        size = position_size(
            balance,
            combined["confidence"],
            market["yes_price"],
            combined["signal"],
        )
        size = min(size, balance)
        if size < 1.0:
            continue

        balance -= size

        # Determine outcome
        resolves_yes = bool(market["resolves_yes"])
        won = (combined["signal"] == "YES" and resolves_yes) or \
              (combined["signal"] == "NO" and not resolves_yes)

        if combined["signal"] == "YES":
            entry_price = market["yes_price"] / 100.0
        else:
            entry_price = (100 - market["yes_price"]) / 100.0

        if won:
            contracts = size / entry_price
            payout = contracts * 1.0
            pnl = payout - size
        else:
            pnl = -size

        balance += size + pnl

        trades.append({
            "ticker": market["ticker"],
            "signal": combined["signal"],
            "edge": combined["edge"],
            "confidence": combined["confidence"],
            "size": size,
            "entry_price_pct": entry_price * 100,
            "true_prob": market["true_prob"],
            "resolves_yes": resolves_yes,
            "won": won,
            "pnl": pnl,
            "balance_after": balance,
        })

    df = pd.DataFrame(trades)
    return _report(df, starting_balance, balance, n_markets, signals_fired)


def _report(df: pd.DataFrame, start: float, end: float, n_markets: int, signals_fired: int) -> dict:
    if df.empty:
        return {"error": "No trades executed"}

    wins = df["won"].sum()
    losses = len(df) - wins
    win_rate = wins / len(df)
    total_pnl = end - start
    roi = total_pnl / start * 100

    avg_win = df.loc[df["won"], "pnl"].mean() if wins > 0 else 0
    avg_loss = df.loc[~df["won"], "pnl"].mean() if losses > 0 else 0
    profit_factor = (wins * avg_win) / abs(losses * avg_loss) if losses > 0 and avg_loss < 0 else float("inf")

    # Max drawdown
    balances = [start] + df["balance_after"].tolist()
    peak = start
    max_dd = 0.0
    for b in balances:
        if b > peak:
            peak = b
        dd = (peak - b) / peak
        max_dd = max(max_dd, dd)

    # Sharpe (daily-equivalent)
    pnl_series = df["pnl"]
    sharpe = (pnl_series.mean() / pnl_series.std() * np.sqrt(252)) if pnl_series.std() > 0 else 0.0

    result = {
        "markets_scanned": n_markets,
        "signals_fired": signals_fired,
        "trades_executed": len(df),
        "signal_rate": f"{signals_fired / n_markets * 100:.1f}%",
        "trade_rate": f"{len(df) / n_markets * 100:.1f}%",
        "starting_balance": start,
        "ending_balance": round(end, 2),
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(roi, 2),
        "win_rate": round(win_rate * 100, 1),
        "wins": int(wins),
        "losses": int(losses),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "avg_edge": round(df["edge"].mean(), 4),
        "avg_confidence": round(df["confidence"].mean(), 4),
    }
    return result, df


def print_report(result: dict, df: pd.DataFrame):
    print("\n" + "=" * 62)
    print("  KALSHI BOT — BACKTEST RESULTS")
    print("=" * 62)
    print(f"  Markets scanned       : {result['markets_scanned']}")
    print(f"  Signals fired         : {result['signals_fired']}  ({result['signal_rate']})")
    print(f"  Trades executed       : {result['trades_executed']}  ({result['trade_rate']})")
    print()
    print(f"  Starting Balance      : ${result['starting_balance']:>10.2f}")
    print(f"  Ending Balance        : ${result['ending_balance']:>10.2f}")
    print(f"  Total P&L             : ${result['total_pnl']:>+10.2f}")
    print(f"  ROI                   : {result['roi_pct']:>+10.2f}%")
    print()
    print(f"  Win Rate              : {result['win_rate']:>9.1f}%")
    print(f"  Wins / Losses         : {result['wins']} / {result['losses']}")
    print(f"  Avg Win               : ${result['avg_win']:>+9.2f}")
    print(f"  Avg Loss              : ${result['avg_loss']:>+9.2f}")
    print(f"  Profit Factor         : {result['profit_factor']:>9.2f}x")
    print()
    print(f"  Max Drawdown          : {result['max_drawdown_pct']:>9.2f}%")
    print(f"  Sharpe Ratio          : {result['sharpe_ratio']:>9.3f}")
    print()
    print(f"  Avg Signal Edge       : {result['avg_edge']:>9.4f}")
    print(f"  Avg Signal Confidence : {result['avg_confidence']:>9.4f}")
    print("=" * 62)

    # Per-signal-source breakdown
    print("\n  Signal Distribution:")
    yes_trades = len(df[df["signal"] == "YES"])
    no_trades = len(df[df["signal"] == "NO"])
    print(f"    YES trades : {yes_trades}")
    print(f"    NO  trades : {no_trades}")

    print("\n  Performance by Entry Price Bucket:")
    df["bucket"] = pd.cut(df["entry_price_pct"], bins=[0, 20, 40, 60, 80, 100])
    grp = df.groupby("bucket", observed=True).agg(
        trades=("pnl", "count"),
        win_rate=("won", "mean"),
        total_pnl=("pnl", "sum"),
    )
    for bucket, row in grp.iterrows():
        print(
            f"    {str(bucket):<18} trades={int(row['trades']):>3}  "
            f"win={row['win_rate']*100:>5.1f}%  pnl=${row['total_pnl']:>+7.2f}"
        )

    # Equity curve (simple ASCII)
    print("\n  Equity Curve (sampled):")
    sampled = df["balance_after"].tolist()
    step = max(1, len(sampled) // 20)
    sampled_pts = [result["starting_balance"]] + sampled[::step]
    min_b, max_b = min(sampled_pts), max(sampled_pts)
    rng_b = max_b - min_b or 1
    for i, b in enumerate(sampled_pts):
        bar_len = int((b - min_b) / rng_b * 40)
        label = f"${b:>8.2f}"
        print(f"    {label} | {'█' * bar_len}")

    print("\n" + "=" * 62 + "\n")


if __name__ == "__main__":
    print("Training/loading ML model on synthetic data...")
    result, df = run_backtest(n_markets=500)
    print_report(result, df)
