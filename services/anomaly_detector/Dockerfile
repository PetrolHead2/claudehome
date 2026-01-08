FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENV REDIS_URL=redis://redis:6379
ENV CLAUDE_MODEL=claude-sonnet-4-20250514

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]