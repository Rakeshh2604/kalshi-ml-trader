FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY kalshi_bot/ ./kalshi_bot/

# Railway mounts a persistent volume at /data
# All DB files and model go there so they survive redeploys
ENV PAPER_DB_PATH=/data/kalshi_paper.db
ENV MODEL_PATH=/data/model.pkl

RUN mkdir -p /data

CMD ["python", "-m", "kalshi_bot.main"]
