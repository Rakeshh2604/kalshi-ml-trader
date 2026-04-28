"""
Performance analytics: Sharpe, Sortino, Calmar, VaR, CVaR,
max drawdown + duration, underwater plot, monthly returns table.
"""

import numpy as np
import pandas as pd
from typing import Sequence


# ── Core metric calculations ──────────────────────────────────────────────────

def sharpe(returns: Sequence[float], risk_free: float = 0.0,
           periods_per_year: float = 252) -> float:
    r = np.array(returns)
    if len(r) < 2 or r.std() == 0:
        return 0.0
    excess = r - risk_free / periods_per_year
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def sortino(returns: Sequence[float], mar: float = 0.0,
            periods_per_year: float = 252) -> float:
    """Sortino ratio: penalises only downside volatility."""
    r = np.array(returns)
    excess = r - mar / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: Sequence[float]) -> tuple[float, int]:
    """
    Returns (max_drawdown_fraction, duration_in_bars).
    duration = longest number of bars spent underwater.
    """
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min())

    # Duration: longest consecutive period below previous peak
    underwater = dd < 0
    max_dur = 0
    cur_dur = 0
    for u in underwater:
        if u:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0

    return max_dd, max_dur


def calmar(equity_curve: Sequence[float],
           periods_per_year: float = 252) -> float:
    """Calmar ratio: CAGR / max drawdown."""
    eq = np.array(equity_curve)
    if len(eq) < 2 or eq[0] == 0:
        return 0.0
    total_return = (eq[-1] - eq[0]) / eq[0]
    n_periods = len(eq)
    cagr = (1 + total_return) ** (periods_per_year / n_periods) - 1
    max_dd, _ = max_drawdown(eq)
    if max_dd == 0:
        return float("inf")
    return float(cagr / abs(max_dd))


def var(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Value at Risk: worst loss at given confidence level (positive = loss)."""
    r = np.array(returns)
    return float(-np.percentile(r, (1 - confidence) * 100))


def cvar(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall): mean loss beyond VaR."""
    r = np.array(returns)
    threshold = np.percentile(r, (1 - confidence) * 100)
    tail = r[r <= threshold]
    if len(tail) == 0:
        return float(-threshold)
    return float(-tail.mean())


def profit_factor(returns: Sequence[float]) -> float:
    r = np.array(returns)
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    return float(gains / losses) if losses > 0 else float("inf")


def full_stats(equity_curve: Sequence[float],
               trade_returns: Sequence[float] | None = None) -> dict:
    """Compute all performance metrics in one call."""
    eq = np.array(equity_curve)
    start = eq[0]
    end = eq[-1]

    # Convert equity to per-trade returns (if not supplied)
    if trade_returns is None or len(trade_returns) == 0:
        trade_returns = list(np.diff(eq) / eq[:-1])

    r = np.array(trade_returns)
    max_dd, dd_dur = max_drawdown(eq)

    return {
        "starting_balance": round(float(start), 2),
        "ending_balance": round(float(end), 2),
        "total_pnl": round(float(end - start), 2),
        "roi_pct": round(float((end - start) / start * 100), 2),
        "total_trades": len(r),
        "wins": int((r > 0).sum()),
        "losses": int((r < 0).sum()),
        "win_rate_pct": round(float((r > 0).mean() * 100), 1),
        "avg_win": round(float(r[r > 0].mean()) if (r > 0).any() else 0.0, 2),
        "avg_loss": round(float(r[r < 0].mean()) if (r < 0).any() else 0.0, 2),
        "profit_factor": round(profit_factor(r), 3),
        "sharpe": round(sharpe(r), 3),
        "sortino": round(sortino(r), 3),
        "calmar": round(calmar(eq), 3),
        "max_drawdown_pct": round(abs(max_dd) * 100, 2),
        "max_dd_duration_trades": dd_dur,
        "var_95_pct": round(var(r) * 100, 2),
        "cvar_95_pct": round(cvar(r) * 100, 2),
        "volatility_pct": round(float(r.std()) * 100, 2),
        "skewness": round(float(pd.Series(r).skew()), 3),
        "kurtosis": round(float(pd.Series(r).kurtosis()), 3),
    }


# ── Monte Carlo simulation ────────────────────────────────────────────────────

def monte_carlo(trade_pnls: Sequence[float], starting_balance: float,
                n_simulations: int = 1000,
                percentiles: tuple[int, ...] = (5, 25, 50, 75, 95)) -> dict:
    """
    Bootstrap resample trade PnLs n_simulations times.
    Returns distribution of final equity, Sharpe, and max drawdown.
    """
    pnls = np.array(trade_pnls)
    n_trades = len(pnls)
    if n_trades == 0:
        return {}

    rng = np.random.default_rng(42)
    final_equities = []
    max_drawdowns = []
    sharpes = []

    for _ in range(n_simulations):
        sample = rng.choice(pnls, size=n_trades, replace=True)
        equity = starting_balance + np.cumsum(sample)
        equity = np.concatenate([[starting_balance], equity])

        final_equities.append(equity[-1])
        max_dd, _ = max_drawdown(equity)
        max_drawdowns.append(abs(max_dd) * 100)

        returns = sample / equity[:-1]
        sharpes.append(sharpe(returns))

    results = {}
    for label, data in [
        ("final_equity", final_equities),
        ("max_drawdown_pct", max_drawdowns),
        ("sharpe", sharpes),
    ]:
        arr = np.array(data)
        results[label] = {
            str(p): round(float(np.percentile(arr, p)), 3)
            for p in percentiles
        }
        results[label]["mean"] = round(float(arr.mean()), 3)

    return results


# ── Walk-forward helper ───────────────────────────────────────────────────────

def walk_forward_summary(fold_results: list[dict]) -> dict:
    """Aggregate fold-level stats into overall walk-forward summary."""
    if not fold_results:
        return {}
    all_pnl = [f["total_pnl"] for f in fold_results]
    all_wr = [f["win_rate_pct"] for f in fold_results]
    all_sharpe = [f["sharpe"] for f in fold_results]
    all_dd = [f["max_drawdown_pct"] for f in fold_results]
    profitable_folds = sum(1 for p in all_pnl if p > 0)
    return {
        "folds": len(fold_results),
        "profitable_folds": profitable_folds,
        "avg_pnl_per_fold": round(float(np.mean(all_pnl)), 2),
        "std_pnl_per_fold": round(float(np.std(all_pnl)), 2),
        "avg_win_rate_pct": round(float(np.mean(all_wr)), 1),
        "avg_sharpe": round(float(np.mean(all_sharpe)), 3),
        "avg_max_drawdown_pct": round(float(np.mean(all_dd)), 2),
        "consistency_score": round(profitable_folds / len(fold_results), 2),
    }


# ── Underwater plot (ASCII) ───────────────────────────────────────────────────

def underwater_plot(equity_curve: Sequence[float], width: int = 50) -> str:
    """Return ASCII underwater (drawdown) chart."""
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100  # negative percentages

    step = max(1, len(dd) // width)
    sampled = dd[::step]

    min_dd = sampled.min()
    if min_dd == 0:
        return "  (No drawdown recorded)"

    lines = []
    for d in sampled:
        bar_len = int(abs(d / min_dd) * 20)
        lines.append(f"  {d:>7.2f}% {'▓' * bar_len}")
    return "\n".join(lines)


# ── Monthly returns heatmap (ASCII) ──────────────────────────────────────────

def monthly_returns_table(trade_df: "pd.DataFrame") -> str:
    """
    Build a simple monthly P&L table from a DataFrame with 'pnl' column.
    Expects trade_df to have a DatetimeIndex or 'timestamp' column.
    """
    if trade_df.empty or "pnl" not in trade_df.columns:
        return "  (No trade data)"

    df = trade_df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    monthly = df["pnl"].resample("ME").sum()
    if monthly.empty:
        return "  (No monthly data)"

    lines = ["  Month        PnL      Bar"]
    for period, val in monthly.items():
        bar_len = int(min(abs(val) / 5, 30))
        bar = ("+" if val >= 0 else "-") * bar_len
        color = "▲" if val >= 0 else "▼"
        lines.append(f"  {period.strftime('%Y-%m')}   ${val:>+8.2f}  {color} {bar}")
    return "\n".join(lines)
