#!/bin/bash
# Quick paper trading status check
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "=========================================="
echo "  PAPER TRADING STATUS"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Show last 30 lines of log
if [ -f "logs/bot.log" ]; then
    echo ""
    echo "--- Recent activity (last 30 lines) ---"
    tail -30 logs/bot.log
else
    echo "No log file yet. Is the bot running?"
fi

# Show DB stats
echo ""
echo "--- Portfolio summary ---"
.venv/bin/python - <<'EOF'
import sqlite3
conn = sqlite3.connect("kalshi_bot/kalshi_paper.db")
conn.row_factory = sqlite3.Row

trades = conn.execute("SELECT * FROM trades ORDER BY timestamp DESC").fetchall()
if not trades:
    print("  No trades recorded yet.")
else:
    closed = [t for t in trades if t["status"] == "closed"]
    open_t = [t for t in trades if t["status"] == "open"]
    wins = [t for t in closed if (t["pnl"] or 0) > 0]
    total_pnl = sum((t["pnl"] or 0) for t in closed)
    print(f"  Total trades    : {len(trades)}")
    print(f"  Open positions  : {len(open_t)}")
    print(f"  Closed trades   : {len(closed)}")
    print(f"  Win rate        : {len(wins)/len(closed)*100:.1f}%" if closed else "  Win rate        : N/A")
    print(f"  Total P&L       : ${total_pnl:+.2f}")
    if open_t:
        print(f"\n  Open positions:")
        for t in open_t:
            print(f"    {t['ticker']:40s} {t['direction']} @ {t['entry_price']}¢  ${t['size']:.2f}")
conn.close()
EOF

echo ""
