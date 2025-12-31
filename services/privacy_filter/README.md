# Privacy Filter Service

FastAPI service that anonymizes and de-anonymizes Home Assistant entity data before sending to Claude API.

## Purpose

Ensures that sensitive information (real entity names, locations, person names) never leaves the local system unprotected.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/anonymize` | POST | Anonymize data before Claude API |
| `/deanonymize` | POST | De-anonymize Claude responses |
| `/reload` | POST | Reload mappings from disk |
| `/mappings/stats` | GET | Get mapping statistics |

## Usage

### Anonymize data
```bash
curl -X POST http://localhost:8001/anonymize \
  -H "Content-Type: application/json" \
  -d '{"data": {"entity_id": "camera.frigate_15", "person": "Magnus"}}'
```

Response:
```json
{
  "data": {"entity_id": "camera.cam_point_a1", "person": "person_1"},
  "mappings_used": 2
}
```

### De-anonymize response
```bash
curl -X POST http://localhost:8001/deanonymize \
  -H "Content-Type: application/json" \
  -d '{"data": {"target": "camera.cam_point_a1"}}'
```

## Configuration

Mappings are loaded from `/opt/claudehome/.claude/skills/entity_mappings.json`.

## Running Locally

```bash
pip install -r requirements.txt
python main.py
```

## Docker

```bash
docker build -t claudehome/privacy-filter .
docker run -p 8001:8001 -v /opt/claudehome/.claude/skills:/opt/claudehome/.claude/skills:ro claudehome/privacy-filter
```
