# Prediction Service Dockerfile
# Builds container with ML dependencies and FastAPI

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# gcc, g++ needed for compiling some Python packages (numpy, pandas)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models/ models/
COPY utils/ utils/
COPY main.py .

# Create directories for model storage
RUN mkdir -p /app/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]