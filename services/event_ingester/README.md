# Event Ingester Service

Monitors Home Assistant state changes and queues them for processing.

## Purpose

Polls the Home Assistant SQLite database for state changes and pushes events to a Redis queue for consumption by other ClaudeHome services.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Ingestion statistics |
| `/queue/length` | GET | Get queue length |
| `/queue/peek` | GET | View events without removing |
| `/queue/clear` | POST | Clear the queue |
| `/reset` | POST | Reset to latest state_id |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HA_CONFIG_DIR` | `/media/pi/NextCloud/homeassistant` | Path to HA config |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `POLL_INTERVAL` | `1.0` | Polling interval in seconds |

## Event Format

Events are pushed to Redis queue `claudehome:events`:

```json
{
  "type": "state_changed",
  "entity_id": "sensor.temperature",
  "state": "23.5",
  "timestamp": "2025-12-31T12:00:00.000000",
  "state_id": 12345,
  "attributes": {"unit_of_measurement": "°C"}
}
```

## Running Locally

```bash
# Start Redis first
docker run -d -p 6379:6379 redis:alpine

# Run the service
pip install -r requirements.txt
HA_CONFIG_DIR=/media/pi/NextCloud/homeassistant python main.py
```

## Docker

```bash
docker build -t claudehome/event-ingester .
docker run -p 8002:8002 \
  -v /media/pi/NextCloud/homeassistant:/config:ro \
  -e HA_CONFIG_DIR=/config \
  -e REDIS_URL=redis://host.docker.internal:6379 \
  claudehome/event-ingester
```
