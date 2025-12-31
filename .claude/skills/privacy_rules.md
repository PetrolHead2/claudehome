# Privacy Rules for ClaudeHome

**Purpose**: Define privacy classification and anonymization rules for Home Assistant data
**Last Updated**: 2025-12-31

---

## 1. Privacy Classification Levels

### Level 1: PUBLIC
Data that can be shared without anonymization.
- Weather data (temperature, humidity, conditions)
- Time-based triggers
- Generic device states (on/off)
- Energy prices (Nordpool)
- System metrics (uptime, performance)

### Level 2: SEMI-PRIVATE
Data that should be anonymized before external processing.
- Energy consumption values (without location context)
- Device power readings
- Automation triggers (anonymized entity IDs)
- Generic motion detection events

### Level 3: PRIVATE
Data that must always be anonymized.
- Camera entity IDs and feeds
- Device tracker locations
- Person names and identities
- Zone names and coordinates
- Phone/device identifiers
- Family member names

### Level 4: SENSITIVE
Data that should never leave the local system.
- API keys and tokens
- Passwords and credentials
- GPS coordinates
- IP addresses
- MAC addresses
- Alarm codes and states

---

## 2. Entity Classification Rules

### By Domain

| Domain | Level | Anonymization Rule |
|--------|-------|-------------------|
| camera | PRIVATE | `camera.{name}` → `camera.cam_point_{hash}` |
| device_tracker | PRIVATE | `device_tracker.{name}` → `device_tracker.tracker_{hash}` |
| person | PRIVATE | `person.{name}` → `person.person_{n}` |
| zone | PRIVATE | `zone.{name}` → `zone.location_{hash}` |
| alarm_control_panel | SENSITIVE | Never expose state details |
| lock | SENSITIVE | Anonymize, never expose codes |
| binary_sensor (motion) | SEMI-PRIVATE | Anonymize location context |
| binary_sensor (door/window) | PRIVATE | Anonymize specific door/window |
| sensor (energy) | SEMI-PRIVATE | Keep values, anonymize source |
| sensor (weather) | PUBLIC | No anonymization needed |
| light | SEMI-PRIVATE | Anonymize room names |
| switch | SEMI-PRIVATE | Anonymize device names |
| media_player | SEMI-PRIVATE | Anonymize room/device names |
| input_text (regions) | PRIVATE | Contains location data |

### By Entity Pattern

```python
PRIVACY_PATTERNS = {
    # Level 4: SENSITIVE - Never expose
    r".*password.*": "SENSITIVE",
    r".*token.*": "SENSITIVE",
    r".*api_key.*": "SENSITIVE",
    r".*secret.*": "SENSITIVE",
    r"alarm_control_panel\..*": "SENSITIVE",

    # Level 3: PRIVATE - Always anonymize
    r"camera\..*": "PRIVATE",
    r"device_tracker\..*": "PRIVATE",
    r"person\..*": "PRIVATE",
    r"zone\..*": "PRIVATE",
    r"input_text\.(magnus|victoria|william|asia)region": "PRIVATE",
    r".*_mobil$": "PRIVATE",
    r".*gps.*": "PRIVATE",
    r".*location.*": "PRIVATE",

    # Level 2: SEMI-PRIVATE - Anonymize identifiers
    r"sensor\..*_power$": "SEMI-PRIVATE",
    r"sensor\..*_energy$": "SEMI-PRIVATE",
    r"binary_sensor\..*motion.*": "SEMI-PRIVATE",
    r"light\..*": "SEMI-PRIVATE",
    r"switch\..*": "SEMI-PRIVATE",
    r"media_player\..*": "SEMI-PRIVATE",

    # Level 1: PUBLIC - No anonymization
    r"sensor\.nordpool.*": "PUBLIC",
    r"sensor\.openweathermap.*": "PUBLIC",
    r"sun\.sun": "PUBLIC",
    r"weather\..*": "PUBLIC",
}
```

---

## 3. Anonymization Strategies

### Hash-Based Anonymization
For entities that need consistent but anonymous identifiers:
```python
import hashlib

def anonymize_entity(entity_id: str, salt: str) -> str:
    """Generate consistent anonymous ID."""
    domain, name = entity_id.split(".", 1)
    hash_input = f"{salt}:{entity_id}"
    hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:8]

    prefix_map = {
        "camera": "cam_point",
        "device_tracker": "tracker",
        "person": "person",
        "zone": "location",
        "binary_sensor": "sensor",
        "light": "light",
        "switch": "switch",
    }
    prefix = prefix_map.get(domain, "entity")
    return f"{domain}.{prefix}_{hash_value}"
```

### Sequential Anonymization
For person entities (consistent numbering):
```python
PERSON_MAP = {
    "person.magnus": "person.person_1",
    "person.victoria": "person.person_2",
    "person.william": "person.person_3",
    "person.asia": "person.person_4",
}
```

### Value Anonymization
For numeric values that reveal patterns:
```python
def anonymize_value(value: float, noise_percent: float = 5) -> float:
    """Add small noise to prevent exact value matching."""
    import random
    noise = value * (random.uniform(-noise_percent, noise_percent) / 100)
    return round(value + noise, 2)
```

---

## 4. Context Stripping Rules

### Location Context
Remove or replace:
- Street names → "primary_residence"
- City names → "metro_area"
- Coordinates → null or centroid of region
- Zone names → "zone_A", "zone_B", etc.

### Temporal Context
For sensitive events:
- Round timestamps to nearest hour
- Remove exact trigger times for security events
- Keep relative time (morning, evening, night)

### Family Context
- Replace names with "person_1", "person_2", etc.
- Replace "magnus_mobil" with "tracker_primary"
- Remove relationship indicators

---

## 5. Data Flow Rules

### Before Claude API
All data MUST pass through privacy filter:
```
Raw Event → Privacy Filter → Anonymized Event → Claude API
```

### Response Processing
Claude responses MUST be de-anonymized:
```
Claude Response → De-anonymizer → Executable Actions
```

### Logging Rules
- NEVER log real entity IDs at DEBUG level
- Use anonymized IDs in all logs
- Sensitive values marked as [REDACTED]

---

## 6. Special Cases

### Car Data (JBB78W)
- License plate: ALWAYS redact
- VIN: ALWAYS redact
- Location history: PRIVATE
- Charging data: SEMI-PRIVATE (values OK, location anonymized)
- Fuel/battery levels: SEMI-PRIVATE

### Camera Data
- Stream URLs: SENSITIVE (never expose)
- Snapshot paths: PRIVATE
- Detection events: SEMI-PRIVATE (anonymize camera ID)
- Person names in detections: PRIVATE

### Alarm System (Sector)
- Arm/disarm states: SENSITIVE
- Zone triggers: PRIVATE
- Codes: SENSITIVE (never in any log)

---

## 7. Compliance Notes

### GDPR Considerations
- Personal data (names, locations) must be anonymizable
- Right to erasure: entity_mappings.json can be deleted
- Data minimization: only collect what's needed

### Data Retention
- Anonymized event cache: 24 hours
- Pattern history: 30 days (anonymized)
- Raw events: Never persist outside HA

---

**Maintained By**: ClaudeHome System
**Review Schedule**: Monthly or after privacy incidents
**Last Reviewed**: 2025-12-31
