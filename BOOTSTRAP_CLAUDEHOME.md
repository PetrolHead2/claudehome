# Task: Bootstrap ClaudeHome System

I am Claude Code, and I need to bootstrap the ClaudeHome autonomous smart home system.

## Phase 1: Self-Awareness
Create `.claude/skills/claude_capabilities.md` documenting:
- My API capabilities as Claude Sonnet 4.5
- How I should be used in ClaudeHome
- Cost optimization strategies
- Best practices for autonomous systems

## Phase 2: Domain Knowledge  
Create `.claude/skills/home_assistant_core.md` with:
- General Home Assistant architecture
- YAML automation syntax reference
- Common patterns and best practices
- Integration types
- Performance optimization

## Phase 3: Installation Analysis
Analyze the Home Assistant installation at `/media/pi/NextCloud/homeassistant/`:
- Read installation_analysis.txt
- Read .storage/core.entity_registry (contains all 1,729 entities)
- Read configuration.yaml
- Read sample of automations.yaml
- Analyze custom_components directory

Create `.claude/skills/my_installation.md` with SPECIFIC details about THIS installation.

## Phase 4: Privacy Configuration
Based on the installation analysis, create:
- `.claude/skills/privacy_rules.md` - Privacy classification rules
- `.claude/skills/entity_mappings.json` - Anonymization mappings

Map sensitive entities:
- cameras ? camera.exterior_point_A, camera.interior_point_B
- device_trackers ? device_tracker.person_1_mobile
- zones ? zone.primary_location
- person entities ? person.person_1

## Phase 5: Generate Core Services
In `/opt/claudehome/services/`, create:

### 5.1 Privacy Filter Service
`privacy_filter/main.py` - FastAPI service that:
- Loads entity_mappings.json
- Provides anonymize() and de_anonymize() endpoints
- Never logs real entity IDs

### 5.2 Event Ingester
`event_ingester/main.py` - Service that:
- Monitors /media/pi/NextCloud/homeassistant/ for changes
- Uses inotifywait for .storage/ changes
- Polls database every 1s for state changes
- Pushes to Redis queue

### 5.3 Health Monitor
`health_monitor/main.py` - Service that:
- Monitors home-assistant.log for errors
- Tracks pattern failures
- Sends Discord alerts
- Suggests auto-fixes

Each service should include:
- main.py (FastAPI application)
- requirements.txt
- Dockerfile
- README.md

## Phase 6: Docker Compose
Create `docker-compose.yml` that:
- Defines all services
- Sets up Redis
- Configures volumes
- Maps to /media/pi/NextCloud/homeassistant

## Execution

Execute these phases in order, creating all files and documenting your decisions.
