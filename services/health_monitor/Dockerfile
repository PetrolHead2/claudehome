FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .

# Create SSH directory and set permissions BEFORE switching user
RUN mkdir -p /app/.ssh && chmod 700 /app/.ssh

# Add this section (FIX):
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8003

ENV HA_CONFIG_DIR=/config
ENV CHECK_INTERVAL=10.0

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
