# Orchestrator Service

Main coordination service for ClaudeHome autonomous smart home system.

## Purpose

The orchestrator connects all ClaudeHome services and implements the event processing pipeline:

```
Redis Queue → Privacy Filter → Triage (Claude) → Routing → Actions
```

## Features

- **Event Triage**: Uses Claude to classify events into categories
- **Batch Processing**: Processes events in batches for efficiency
- **Privacy-First**: All events anonymized before Claude API calls
- **Service Health Monitoring**: Tracks health of all connected services
- **Critical Event Handling**: Immediate handling of critical events

## Event Categories

| Category | Description | Action |
|----------|-------------|--------|
| ROUTINE | Normal expected behavior | Ignore |
| ANOMALY | Unusual pattern | Log, analyze |
| PROACTIVE | Automation opportunity | Suggest |
| CAUSAL | User override detected | Learn |
| CRITICAL | Immediate attention needed | Alert |

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with service status |
| `/stats` | GET | Processing statistics |
| `/triage/recent` | GET | Recent triage results |
| `/triage/manual` | POST | Manually triage an event |
| `/critical` | GET | Critical events queue |
| `/reset/stats` | POST | Reset statistics |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | `` | Anthropic API key (required) |
| `REDIS_URL` | `redis://redis:6379` | Redis connection URL |
| `PRIVACY_FILTER_URL` | `http://privacy_filter:8001` | Privacy filter service URL |
| `HEALTH_MONITOR_URL` | `http://health_monitor:8003` | Health monitor URL |
| `EVENT_INGESTER_URL` | `http://event_ingester:8002` | Event ingester URL |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `BATCH_SIZE` | `10` | Events per batch |
| `BATCH_WAIT_SECONDS` | `5.0` | Wait time between batches |

## Triage System Prompt

The orchestrator uses a specialized prompt for event triage:

```
Categories:
- ROUTINE: Normal expected behavior
- ANOMALY: Unusual pattern
- PROACTIVE: Automation opportunity
- CAUSAL: User override detected
- CRITICAL: Immediate attention needed
```

Claude responds with structured JSON including category, confidence, reasoning, and suggested actions.

## Running Locally

```bash
pip install -r requirements.txt
ANTHROPIC_API_KEY=sk-ant-xxx \
REDIS_URL=redis://localhost:6379 \
python main.py
```

## Docker

```bash
docker build -t claudehome/orchestrator .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-xxx \
  -e REDIS_URL=redis://host.docker.internal:6379 \
  claudehome/orchestrator
```

## Cost Optimization

The orchestrator implements several cost-saving measures:

1. **Event Filtering**: Skips obviously routine events (power sensors, signal strength)
2. **Batch Processing**: Groups events to reduce API overhead
3. **Prompt Caching**: System prompt is cached by Anthropic
4. **Concise Responses**: Limited to 500 tokens per triage

Estimated costs:
- ~$0.0002 per triage (with caching)
- ~$0.20/day at 1000 events/day
