FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

EXPOSE 8000

ENV REDIS_URL=redis://redis:6379
ENV PRIVACY_FILTER_URL=http://privacy_filter:8001
ENV HEALTH_MONITOR_URL=http://health_monitor:8003
ENV EVENT_INGESTER_URL=http://event_ingester:8002
ENV CLAUDE_MODEL=claude-sonnet-4-20250514
ENV BATCH_SIZE=10
ENV BATCH_WAIT_SECONDS=5.0

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
