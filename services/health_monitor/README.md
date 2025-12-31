# Health Monitor Service

Monitors Home Assistant logs for errors and sends alerts.

## Purpose

Watches the Home Assistant log file for errors, classifies them by pattern, tracks frequency, and sends Discord alerts for critical issues.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Monitoring statistics |
| `/errors/recent` | GET | Recent errors |
| `/patterns` | GET | Error pattern statistics |
| `/patterns/reset` | POST | Reset pattern counts |
| `/test/alert` | POST | Send test Discord alert |
| `/suggestions` | GET | Get auto-fix suggestions |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HA_CONFIG_DIR` | `/media/pi/NextCloud/homeassistant` | Path to HA config |
| `DISCORD_WEBHOOK` | `` | Discord webhook URL for alerts |
| `CHECK_INTERVAL` | `10.0` | Log check interval in seconds |

## Error Patterns

The service classifies errors into these patterns:
- `connection_refused` - Network connectivity issues
- `timeout` - Timeout errors
- `authentication` - Auth failures (401/403)
- `integration_setup` - Integration setup failures
- `entity_unavailable` - Unavailable entities
- `database` - Database lock/errors
- `memory` - Memory issues
- `template` - Template errors
- `yaml` - YAML configuration errors
- `api` - API errors

## Discord Alerts

Alerts are sent for:
- Any CRITICAL level errors
- Patterns that occur more than 5 times

Alert includes:
- Error level and component
- Classified pattern
- Error message
- Auto-fix suggestion (if available)

## Running Locally

```bash
pip install -r requirements.txt
HA_CONFIG_DIR=/media/pi/NextCloud/homeassistant \
DISCORD_WEBHOOK=https://discord.com/api/webhooks/... \
python main.py
```

## Docker

```bash
docker build -t claudehome/health-monitor .
docker run -p 8003:8003 \
  -v /media/pi/NextCloud/homeassistant:/config:ro \
  -e HA_CONFIG_DIR=/config \
  -e DISCORD_WEBHOOK=https://discord.com/api/webhooks/... \
  claudehome/health-monitor
```
