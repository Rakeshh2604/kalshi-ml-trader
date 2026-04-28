import os
from dotenv import load_dotenv

load_dotenv()

# API credentials
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Kalshi API endpoint
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Trading settings
SCAN_INTERVAL_SECONDS = 60
MIN_LIQUIDITY = 1000          # minimum volume in contracts
MIN_EDGE_THRESHOLD = 0.05     # minimum edge to consider a trade
MAX_POSITION_SIZE = 50        # max dollars per trade
STARTING_PAPER_BALANCE = 1000.0
MIN_TIME_TO_CLOSE_HOURS = 1.0   # skip markets closing within this window
MAX_TIME_TO_CLOSE_HOURS = 72.0  # skip markets closing more than 3 days away

# Signal weights (must sum to 1.0)
WEIGHT_MISPRICING = 0.40
WEIGHT_SENTIMENT = 0.30
WEIGHT_ML = 0.30

# Risk management
MAX_KELLY_FRACTION = 0.25     # quarter Kelly cap
MIN_CONFIDENCE = 0.60         # minimum confidence to trade

# ML model
MODEL_PATH = "kalshi_bot/model.pkl"
SYNTHETIC_TRAIN_SAMPLES = 2000

# News cache TTL (seconds)
NEWS_CACHE_TTL = 900  # 15 minutes

# Paper trading flag — never change to False without full audit
PAPER_TRADING = True
