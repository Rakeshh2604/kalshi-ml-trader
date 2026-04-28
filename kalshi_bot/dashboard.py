"""
Rich CLI dashboard — run as: python -m kalshi_bot.dashboard
"""

import numpy as np
from datetime import datetime, timezone
from kalshi_bot import database as db, analytics
from kalshi_bot.config import STARTING_PAPER_BALANCE


def _equity_curve_ascii(balances: list[float], width: int = 44) -> list[str]:
    if len(balances) < 2:
        return ["  (not enough data yet)"]
    arr = np.array(balances)
    min_b, max_b = arr.min(), arr.max()
    rng = max_b - min_b or 1.0
    step = max(1, len(arr) // 20)
    sampled = arr[::step]
    start = balances[0]
    lines = []
    for b in sampled:
        bar_len = int((b - min_b) / rng * width)
        marker = "▲" if b >= start else "▼"
        lines.append(f"    {marker} ${b:>9.2f} │{'█' * bar_len}")
    return lines


def _underwater_ascii(balances: list[float], width: int = 36) -> list[str]:
    if len(balances) < 2:
        return ["  (not enough data yet)"]
    arr = np.array(balances)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak * 100
    step = max(1, len(dd) // 15)
    sampled = dd[::step]
    min_dd = sampled.min()
    if min_dd == 0:
        return ["  (no drawdown recorded)"]
    lines = []
    for d in sampled:
        bar_len = int(abs(d / min_dd) * 22) if min_dd != 0 else 0
        lines.append(f"    {d:>7.2f}% {'▓' * bar_len}")
    return lines


def print_dashboard():
    all_trades = db.get_trade_history(limit=500)
    stats = db.get_performance_stats()
    recent = db.get_trade_history(limit=5)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    balance_est = STARTING_PAPER_BALANCE + stats["total_pnl"]

    # Reconstruct equity curve from trade history
    closed = [t for t in reversed(all_trades) if t["status"] == "closed" and t["pnl"] is not None]
    equity_curve = [STARTING_PAPER_BALANCE]
    running = STARTING_PAPER_BALANCE
    trade_pnls = []
    for t in closed:
        running += t["pnl"]
        equity_curve.append(running)
        trade_pnls.append(t["pnl"])

    # Compute advanced metrics
    if len(equity_curve) > 1 and trade_pnls:
        trade_returns = [p / e for p, e in zip(trade_pnls, equity_curve[:-1]) if e != 0]
        adv = analytics.full_stats(equity_curve, trade_returns)
    else:
        adv = {}

    bar = "═" * 64

    print(f"\n{bar}")
    print(f"  {'KALSHI PAPER TRADING DASHBOARD':^60}")
    print(f"  {now:^60}")
    print(bar)

    # ── Portfolio snapshot ─────────────────────────────────────────
    pnl_arrow = "▲" if stats["total_pnl"] >= 0 else "▼"
    print(f"\n  Paper Balance        : ${balance_est:>10.2f}")
    print(f"  Starting Balance     : ${STARTING_PAPER_BALANCE:>10.2f}")
    print(f"  Total P&L            : {pnl_arrow} ${stats['total_pnl']:>+9.2f}")
    print(f"  ROI                  : {(stats['total_pnl']/STARTING_PAPER_BALANCE*100):>+9.2f}%")

    # ── Risk-adjusted metrics ──────────────────────────────────────
    print(f"\n  {'─'*56}")
    print(f"  Risk-Adjusted Metrics")
    print(f"  {'─'*56}")
    print(f"  Sharpe Ratio         : {adv.get('sharpe', 0):>9.3f}")
    print(f"  Sortino Ratio        : {adv.get('sortino', 0):>9.3f}")
    print(f"  Calmar Ratio         : {adv.get('calmar', 0):>9.3f}")
    print(f"  Max Drawdown         : {adv.get('max_drawdown_pct', 0):>9.2f}%")
    print(f"  VaR (95%)            : {adv.get('var_95_pct', 0):>9.2f}%  per trade")
    print(f"  CVaR (95%)           : {adv.get('cvar_95_pct', 0):>9.2f}%  per trade")
    print(f"  Profit Factor        : {adv.get('profit_factor', 0):>9.3f}×")

    # ── Trade stats ────────────────────────────────────────────────
    print(f"\n  {'─'*56}")
    print(f"  Trade Statistics")
    print(f"  {'─'*56}")
    print(f"  Total Trades         : {stats['total_trades']:>10}")
    print(f"  Win Rate             : {stats['win_rate'] * 100:>9.1f}%")
    print(f"  Wins / Losses        : {stats['wins']} / {stats['losses']}")
    print(f"  Avg Win              : ${adv.get('avg_win', 0):>+9.2f}")
    print(f"  Avg Loss             : ${adv.get('avg_loss', 0):>+9.2f}")

    # ── Open positions ─────────────────────────────────────────────
    open_trades = [t for t in all_trades if t["status"] == "open"]
    print(f"\n  {'─'*56}")
    print(f"  Open Positions ({len(open_trades)})")
    print(f"  {'─'*56}")
    if open_trades:
        for t in open_trades:
            print(
                f"  {t['ticker']:<28} {t['direction']:<4} "
                f"${t['size']:>6.2f} @ {t['entry_price']:.0f}¢"
            )
    else:
        print("  (none)")

    # ── Last 5 trades ──────────────────────────────────────────────
    print(f"\n  {'─'*56}")
    print(f"  Last 5 Trades")
    print(f"  {'─'*56}")
    if recent:
        for t in recent:
            pnl_str = f"${t['pnl']:>+6.2f}" if t["pnl"] is not None else "  open"
            marker = "✓" if (t["pnl"] or 0) > 0 else ("✗" if (t["pnl"] or 0) < 0 else "·")
            print(
                f"  {marker} {t['ticker']:<28} {t['direction']:<4} "
                f"${t['size']:>6.2f}  {pnl_str}  [{t['status']}]"
            )
    else:
        print("  (no trades yet)")

    # ── Equity curve ───────────────────────────────────────────────
    if len(equity_curve) > 2:
        print(f"\n  {'─'*56}")
        print(f"  Equity Curve")
        print(f"  {'─'*56}")
        for line in _equity_curve_ascii(equity_curve):
            print(line)

        print(f"\n  Underwater Drawdown")
        print(f"  {'─'*40}")
        for line in _underwater_ascii(equity_curve):
            print(line)

    print(f"\n{bar}\n")


if __name__ == "__main__":
    print_dashboard()
