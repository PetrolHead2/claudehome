# ClaudeHome

Autonomous, privacy-first smart home intelligence system for Home Assistant.

## Overview

ClaudeHome uses Claude AI to provide intelligent automation suggestions, anomaly detection, and system health monitoring for your Home Assistant installation. All data is anonymized before leaving your local network.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Home Assistant                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   States    │  │   Events    │  │  home-assistant.log     │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ClaudeHome Services                         │
│                                                                  │
│  ┌────────────────┐    ┌────────────────┐   ┌────────────────┐  │
│  │ Event Ingester │───▶│     Redis      │◀──│ Health Monitor │  │
│  │   (port 8002)  │    │  (port 6379)   │   │   (port 8003)  │  │
│  └────────────────┘    └───────┬────────┘   └────────────────┘  │
│                                │                                 │
│                                ▼                                 │
│  ┌────────────────┐    ┌────────────────┐                       │
│  │ Privacy Filter │◀───│  Orchestrator  │───▶ Claude API        │
│  │   (port 8001)  │    │   (port 8000)  │    (Anonymized)       │
│  └────────────────┘    └────────────────┘                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Event Triage**: Classifies HA events as routine, anomaly, proactive opportunity, or critical
- **Privacy-First**: All data anonymized before external API calls
- **Health Monitoring**: Watches logs for errors, sends Discord alerts with fix suggestions
- **Cost Optimized**: Batched processing, prompt caching, smart filtering

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Home Assistant installation
- Anthropic API key

### Installation

1. Clone the repository:
```bash
cd /opt
git clone https://github.com/PetrolHead2/claudehome.git
cd claudehome
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your values:
#   ANTHROPIC_API_KEY=sk-ant-xxx
#   DISCORD_WEBHOOK=https://discord.com/api/webhooks/xxx (optional)
```

3. Start services:
```bash
docker-compose up -d
```

4. Verify health:
```bash
curl http://localhost:8000/health
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Orchestrator | 8000 | Main coordination, event triage |
| Privacy Filter | 8001 | Anonymizes/de-anonymizes data |
| Event Ingester | 8002 | Monitors HA state changes |
| Health Monitor | 8003 | Log monitoring, alerts |
| Redis | 6379 | Message queue |

## API Endpoints

### Orchestrator (8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health + dependencies |
| `/stats` | GET | Processing statistics |
| `/triage/recent` | GET | Recent triage results |
| `/triage/manual` | POST | Manually triage an event |
| `/critical` | GET | Critical events queue |

### Privacy Filter (8001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/anonymize` | POST | Anonymize entity data |
| `/deanonymize` | POST | Restore real entity IDs |
| `/mappings/stats` | GET | Mapping statistics |

### Event Ingester (8002)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stats` | GET | Ingestion statistics |
| `/queue/length` | GET | Redis queue length |
| `/queue/peek` | GET | Preview queued events |

### Health Monitor (8003)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stats` | GET | Error statistics |
| `/errors/recent` | GET | Recent errors |
| `/patterns` | GET | Error pattern counts |
| `/test/alert` | POST | Send test Discord alert |

## Configuration

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-xxx

# Optional
DISCORD_WEBHOOK=https://discord.com/api/webhooks/xxx
HA_CONFIG_DIR=/media/pi/NextCloud/homeassistant
```

### Skills (Knowledge Base)

Skills are markdown/JSON files in `.claude/skills/` that provide context:

| File | Purpose |
|------|---------|
| `claude_capabilities.md` | Claude model capabilities and best practices |
| `home_assistant_core.md` | HA architecture and automation reference |
| `my_installation.md` | Specific details about YOUR installation |
| `privacy_rules.md` | Privacy classification rules |
| `entity_mappings.json` | Entity anonymization mappings |

## Event Categories

The orchestrator classifies events into these categories:

| Category | Description | Action |
|----------|-------------|--------|
| **ROUTINE** | Normal expected behavior | Ignore |
| **ANOMALY** | Unusual pattern | Log, investigate |
| **PROACTIVE** | Automation opportunity | Suggest improvement |
| **CAUSAL** | User override detected | Learn preference |
| **CRITICAL** | Immediate attention | Alert, remediate |

## Privacy

ClaudeHome is designed with privacy as a core principle:

1. **Local Processing**: Event ingestion and initial filtering happen locally
2. **Anonymization**: All entity IDs, names, and locations are anonymized before API calls
3. **No Raw Data**: Camera feeds, exact locations, and personal names never leave your network
4. **Configurable**: Customize privacy rules in `privacy_rules.md`

### Anonymization Examples

| Real Entity | Anonymized |
|-------------|------------|
| `person.magnus` | `person.person_1` |
| `camera.frigate_15` | `camera.cam_point_a1` |
| `device_tracker.magnus_mobil` | `device_tracker.tracker_primary` |
| `zone.home` | `zone.location_primary` |

## Cost Estimation

With default settings and typical usage:

| Metric | Estimate |
|--------|----------|
| Events/day | ~1,000-5,000 |
| Triage calls/day | ~100-500 (after filtering) |
| Tokens/triage | ~800 (input) + 200 (output) |
| Cost/day | ~$0.10-0.50 |
| Cost/month | ~$3-15 |

Cost optimization features:
- Event filtering (skips routine sensor updates)
- Batch processing (reduces API overhead)
- Prompt caching (90% reduction on system prompt)
- Concise responses (500 token limit)

## Development

### Running Locally

```bash
# Start Redis
docker run -d -p 6379:6379 redis:alpine

# Run individual service
cd services/orchestrator
pip install -r requirements.txt
ANTHROPIC_API_KEY=xxx python main.py
```

### Building Images

```bash
docker-compose build
```

### Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f orchestrator
```

## Troubleshooting

### Services not starting

```bash
# Check service health
docker-compose ps
docker-compose logs <service-name>
```

### Redis connection failed

```bash
# Verify Redis is running
docker-compose exec redis redis-cli ping
```

### No events being processed

```bash
# Check event ingester
curl http://localhost:8002/stats

# Verify database path
curl http://localhost:8002/health
```

### Claude API errors

```bash
# Verify API key
curl http://localhost:8000/health

# Check orchestrator logs
docker-compose logs orchestrator
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

- GitHub Issues: https://github.com/PetrolHead2/claudehome/issues
- Discord: [Coming soon]
