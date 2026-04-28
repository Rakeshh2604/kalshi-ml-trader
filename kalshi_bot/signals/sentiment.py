"""
News sentiment signal using VADER (local, no API key) + NewsAPI.
Results are cached per market title for NEWS_CACHE_TTL seconds.
"""

import time
import logging
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from kalshi_bot.config import NEWS_API_KEY, NEWS_CACHE_TTL

logger = logging.getLogger(__name__)

_vader = SentimentIntensityAnalyzer()
_cache: dict[str, tuple[float, dict]] = {}  # title -> (timestamp, result)

NONE_SIGNAL = {"signal": "NONE", "edge": 0.0, "confidence": 0.0}


def _fetch_headlines(query: str, max_results: int = 5) -> list[str]:
    if not NEWS_API_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query[:100],
        "pageSize": max_results,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWS_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [a["title"] for a in articles if a.get("title")]
    except Exception as exc:
        logger.warning(f"NewsAPI error for '{query}': {exc}")
        return []


def _score_headlines(headlines: list[str]) -> float:
    if not headlines:
        return 0.0
    scores = [_vader.polarity_scores(h)["compound"] for h in headlines]
    return sum(scores) / len(scores)


def analyze(market: dict) -> dict:
    title = market.get("title", "")
    if not title:
        return NONE_SIGNAL

    # Check cache
    if title in _cache:
        ts, cached_result = _cache[title]
        if time.time() - ts < NEWS_CACHE_TTL:
            return cached_result

    # Build a short search query from the market title
    query_words = title.split()[:6]
    query = " ".join(query_words)

    headlines = _fetch_headlines(query)
    compound = _score_headlines(headlines)

    if not headlines or abs(compound) < 0.05:
        result = NONE_SIGNAL
    else:
        # Map compound [-1, 1] → signal + edge
        edge = min(abs(compound) * 0.20, 0.20)
        confidence = 0.50 + min(abs(compound) * 0.25, 0.35)
        signal = "YES" if compound > 0 else "NO"
        result = {"signal": signal, "edge": edge, "confidence": confidence}

    _cache[title] = (time.time(), result)
    logger.debug(f"[Sentiment] '{title[:40]}' compound={compound:.3f} → {result['signal']}")
    return result
