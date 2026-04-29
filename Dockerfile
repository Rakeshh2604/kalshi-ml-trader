FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY kalshi_bot/ ./kalshi_bot/
COPY kalshi_private_key.pem ./kalshi_private_key.pem

ENV KALSHI_PRIVATE_KEY_PATH=/app/kalshi_private_key.pem

CMD ["python", "-m", "kalshi_bot.main"]
