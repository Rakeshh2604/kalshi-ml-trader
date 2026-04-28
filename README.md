# Kalshi Paper Trading Bot

An algorithmic paper trading bot for Kalshi prediction markets.  
**All trading is simulated — no real money is ever at risk.**

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Description |
|---|---|
| `KALSHI_API_KEY_ID` | Your Kalshi API key ID |
| `KALSHI_PRIVATE_KEY_PATH` | Absolute path to your RSA private key `.pem` file |
| `NEWS_API_KEY` | NewsAPI.org key (free tier works) |

#### Getting your Kalshi API key
1. Log in to [kalshi.com](https://kalshi.com)
2. Go to **Account → Settings → API Keys**
3. Click **Create API Key**
4. Download the private key `.pem` file — store it somewhere safe
5. Copy the Key ID into your `.env`

#### Getting a NewsAPI key
1. Sign up at [newsapi.org](https://newsapi.org) (free)
2. Copy your API key into `.env`

---

## Running a Backtest

```bash
cd "kalshi all bot"
python -m kalshi_bot.backtest
```

The backtest simulates 500 synthetic markets, runs all three signal generators, applies Kelly-criterion position sizing, and prints a full performance report.

---

## Starting the Paper Trading Bot

```bash
python -m kalshi_bot.main
```

The bot will:
1. Train/load the ML model
2. Scan Kalshi every 60 seconds
3. Generate signals (mispricing + sentiment + ML)
4. Place simulated paper trades when edge > 5% and confidence > 60%
5. Auto-close positions when markets expire

---

## Viewing the Dashboard

In a separate terminal:

```bash
python -m kalshi_bot.dashboard
```

---

## Architecture

```
main.py          → orchestration loop
scanner.py       → fetches & filters active markets
signals/
  mispricing.py  → z-score statistical edge detection
  sentiment.py   → VADER sentiment from NewsAPI headlines
  ml_model.py    → XGBoost classifier (trains on synthetic data)
risk.py          → Kelly criterion + position sizing
paper_trader.py  → simulated execution + portfolio tracking
database.py      → SQLite logging (trades + signals)
dashboard.py     → CLI performance monitor
backtest.py      → offline simulation engine
```

---

## Safety

- `PAPER_TRADING = True` is hardcoded in `config.py`
- `place_order()` in `client.py` never calls the real Kalshi API
- Every execution point is marked `# PAPER TRADING - NO REAL ORDERS`
