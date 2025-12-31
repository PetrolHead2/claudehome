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

EXPOSE 8002

ENV HA_CONFIG_DIR=/config
ENV REDIS_URL=redis://redis:6379
ENV POLL_INTERVAL=1.0

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
