# Home Assistant Core Knowledge

**Purpose**: Reference for Home Assistant architecture, syntax, and best practices  
**Target Version**: 2025.x (current installation: 2025.11.3)  
**Last Updated**: 2025-01-31

---

## 1. Architecture Overview

### Event-Driven Core

Home Assistant operates on an **event bus** architecture:
```
Event Bus (Central)
    ?
State Machine (Tracks all entity states)
    ?
Service Registry (Available actions)
    ?
Automations/Scripts (React to events/states)
```

**Key Concepts**:
- **Events**: Things that happen (state_changed, automation_triggered, service_called)
- **States**: Current status of entities (on/off, 23.5°C, locked/unlocked)
- **Services**: Actions you can perform (turn_on, set_temperature, notify)
- **Entities**: Represent devices, sensors, or virtual objects

### State Machine

Every entity has:
- `entity_id`: Unique identifier (domain.object_id)
- `state`: Current value (on, 23.5, locked, etc.)
- `attributes`: Additional metadata (brightness, temperature, battery, etc.)
- `last_changed`: When state value changed
- `last_updated`: When state or attributes changed

### Integration Framework

**Integration Types**:
- **Local Push**: MQTT, webhooks, ESPHome (best performance)
- **Local Poll**: HTTP, REST, scraping (configurable intervals)
- **Cloud API**: OAuth-based services (rate-limited)
- **Custom Components**: User-developed integrations (HACS)

---

## 2. YAML Automation Syntax

### Basic Structure
```yaml
- id: unique_automation_id
  alias: "Human Readable Name"
  description: "What this automation does"
  
  trigger:
    # When to run
    
  condition:
    # Check before running (optional)
    
  action:
    # What to do
    
  mode: single|restart|queued|parallel
```

### Triggers (When to Run)

**State Trigger** (most common):
```yaml
trigger:
  - platform: state
    entity_id: binary_sensor.motion_living_room
    from: "off"
    to: "on"
    for:  # Optional: require state duration
      hours: 0
      minutes: 5
      seconds: 0
```

**Time Trigger**:
```yaml
trigger:
  - platform: time
    at: "07:00:00"
  
  # Or time pattern (every X)
  - platform: time_pattern
    minutes: "/15"  # Every 15 minutes
```

**Numeric State Trigger**:
```yaml
trigger:
  - platform: numeric_state
    entity_id: sensor.temperature
    above: 25
    below: 30
    for: "00:05:00"  # For 5 minutes
```

**Template Trigger** (advanced):
```yaml
trigger:
  - platform: template
    value_template: >
      {{ states('sensor.power') | float > 1000 and 
         is_state('switch.heater', 'on') }}
```

**Event Trigger**:
```yaml
trigger:
  - platform: event
    event_type: automation_triggered
    event_data:
      entity_id: automation.specific_auto
```

**Webhook Trigger** (external systems):
```yaml
trigger:
  - platform: webhook
    webhook_id: my_webhook_123
```

### Conditions (Check Before Running)

**State Condition**:
```yaml
condition:
  - condition: state
    entity_id: light.living_room
    state: "off"
```

**Numeric State Condition**:
```yaml
condition:
  - condition: numeric_state
    entity_id: sensor.temperature
    below: 20
```

**Time Condition**:
```yaml
condition:
  - condition: time
    after: "18:00:00"
    before: "23:00:00"
    weekday:
      - mon
      - tue
      - wed
      - thu
      - fri
```

**Template Condition** (advanced):
```yaml
condition:
  - condition: template
    value_template: >
      {{ states('sensor.power') | float > 100 }}
```

**Zone Condition**:
```yaml
condition:
  - condition: zone
    entity_id: person.john
    zone: zone.home
```

**Logical Operators**:
```yaml
condition:
  - condition: and  # All must be true
    conditions:
      - condition: state
        entity_id: sun.sun
        state: "below_horizon"
      - condition: state
        entity_id: person.john
        state: "home"
```
```yaml
condition:
  - condition: or  # At least one must be true
    conditions:
      - condition: state
        entity_id: binary_sensor.motion_1
        state: "on"
      - condition: state
        entity_id: binary_sensor.motion_2
        state: "on"
```

### Actions (What to Do)

**Service Call**:
```yaml
action:
  - service: light.turn_on
    target:
      entity_id: light.living_room
    data:
      brightness_pct: 50
      transition: 2
```

**Multiple Targets**:
```yaml
action:
  - service: light.turn_on
    target:
      area_id: living_room  # All lights in area
    data:
      brightness_pct: 75
```

**Delay**:
```yaml
action:
  - delay: "00:00:30"  # 30 seconds
  # Or dynamic:
  - delay:
      seconds: "{{ states('input_number.delay') | int }}"
```

**Wait for Trigger**:
```yaml
action:
  - wait_for_trigger:
      - platform: state
        entity_id: binary_sensor.motion
        to: "off"
    timeout: "00:05:00"  # Give up after 5 minutes
    continue_on_timeout: false
```

**Choose (If-Then-Else)**:
```yaml
action:
  - choose:
      # If condition 1
      - conditions:
          - condition: state
            entity_id: sun.sun
            state: "above_horizon"
        sequence:
          - service: light.turn_on
            data:
              brightness_pct: 100
      
      # If condition 2
      - conditions:
          - condition: state
            entity_id: sun.sun
            state: "below_horizon"
        sequence:
          - service: light.turn_on
            data:
              brightness_pct: 30
    
    # Else (default)
    default:
      - service: light.turn_off
```

**Repeat**:
```yaml
action:
  - repeat:
      count: 3
      sequence:
        - service: light.toggle
        - delay: "00:00:01"
```

**Parallel Actions** (simultaneous):
```yaml
action:
  - parallel:
      - service: light.turn_on
        target:
          entity_id: light.bedroom
      - service: climate.set_temperature
        target:
          entity_id: climate.bedroom
        data:
          temperature: 21
```

**Variables**:
```yaml
action:
  - variables:
      my_brightness: 50
  - service: light.turn_on
    data:
      brightness_pct: "{{ my_brightness }}"
```

### Automation Modes
```yaml
mode: single  # Default: Don't run if already running
mode: restart  # Restart automation if triggered again
mode: queued  # Queue up triggers
mode: parallel  # Run multiple instances simultaneously
```

---

## 3. Templates (Jinja2)

### Basic Syntax
```yaml
# State access
{{ states('sensor.temperature') }}

# State with default
{{ states('sensor.temperature') | float(0) }}

# Attributes
{{ state_attr('light.living_room', 'brightness') }}

# Numeric operations
{{ states('sensor.power') | float * 0.001 }}  # W to kW

# Conditionals
{% if is_state('sun.sun', 'above_horizon') %}
  daytime
{% else %}
  nighttime
{% endif %}

# Time functions
{{ now().hour }}
{{ as_timestamp(now()) }}
{{ states.sensor.temperature.last_changed }}
```

### Common Template Functions
```yaml
# State checks
is_state('entity_id', 'state')
is_state_attr('entity_id', 'attr', 'value')

# Lists
states.sensor  # All sensor entities
states.light | selectattr('state', 'eq', 'on') | list

# Time
now()
utcnow()
as_timestamp(states.sensor.time.last_changed)

# Math
float(value)
int(value)
round(value, decimals)

# String
lower()
upper()
replace('old', 'new')
```

---

## 4. Common Automation Patterns

### Motion-Activated Lights
```yaml
- id: motion_lights_living_room
  alias: "Motion ? Living Room Lights"
  trigger:
    - platform: state
      entity_id: binary_sensor.motion_living_room
      to: "on"
  condition:
    - condition: state
      entity_id: sun.sun
      state: "below_horizon"
    - condition: state
      entity_id: light.living_room
      state: "off"
  action:
    - service: light.turn_on
      target:
        entity_id: light.living_room
      data:
        brightness_pct: 60
        transition: 1
    - wait_for_trigger:
        - platform: state
          entity_id: binary_sensor.motion_living_room
          to: "off"
          for: "00:05:00"
    - service: light.turn_off
      target:
        entity_id: light.living_room
      data:
        transition: 2
  mode: restart
```

### Presence-Based Climate
```yaml
- id: climate_presence_away
  alias: "Climate: Away Mode"
  trigger:
    - platform: state
      entity_id: person.john
      from: "home"
      to: "not_home"
      for: "00:30:00"
  action:
    - service: climate.set_temperature
      target:
        entity_id: climate.living_room
      data:
        temperature: 18  # Energy saving
```

### Notification on Anomaly
```yaml
- id: notify_temp_spike
  alias: "Notify: Temperature Spike"
  trigger:
    - platform: numeric_state
      entity_id: sensor.temperature
      above: 30
      for: "00:05:00"
  action:
    - service: notify.mobile_app
      data:
        title: "Temperature Alert"
        message: "Living room temp is {{ states('sensor.temperature') }}°C"
        data:
          priority: high
```

### Time-Based Routine
```yaml
- id: morning_routine
  alias: "Morning Routine"
  trigger:
    - platform: time
      at: "07:00:00"
  condition:
    - condition: state
      entity_id: person.john
      state: "home"
    - condition: state
      entity_id: input_boolean.vacation_mode
      state: "off"
  action:
    - parallel:
        - service: light.turn_on
          target:
            area_id: bedroom
          data:
            brightness_pct: 30
            transition: 300  # 5 minutes
        - service: climate.set_temperature
          target:
            entity_id: climate.bedroom
          data:
            temperature: 21
        - service: cover.open_cover
          target:
            entity_id: cover.bedroom_blinds
```

---

## 5. Performance Best Practices

### Template Efficiency

**Bad** (triggers on every state change):
```yaml
sensor:
  - platform: template
    sensors:
      expensive_sensor:
        value_template: "{{ now() }}"  # Updates every second!
```

**Good** (trigger-based):
```yaml
template:
  - trigger:
      - platform: time_pattern
        minutes: "/5"  # Every 5 minutes
    sensor:
      - name: "Efficient Sensor"
        state: "{{ now() }}"
```

### Recorder Optimization

**Exclude High-Frequency Entities**:
```yaml
recorder:
  purge_keep_days: 30
  exclude:
    entity_globs:
      - sensor.*_rssi  # WiFi signal
      - sensor.*_linkquality  # Zigbee link
      - sensor.*_uptime
    entities:
      - sensor.time
      - sensor.date
```

### Automation Organization

**Use Separate Files**:
```yaml
# configuration.yaml
automation: !include_dir_merge_list automations/

# automations/
#   lights.yaml
#   climate.yaml
#   security.yaml
```

**Use Descriptive IDs**:
```yaml
# Bad
- id: '1725306210562'

# Good
- id: motion_lights_living_room_001
```

---

## 6. Debugging & Troubleshooting

### Template Testing

1. Developer Tools ? Template
2. Paste template
3. See result instantly

### Automation Tracing

1. Settings ? Automations & Scenes
2. Click automation
3. Click "Traces" tab
4. See execution history with timing

### Log Analysis
```yaml
# Enable debug logging for component
logger:
  default: info
  logs:
    homeassistant.components.mqtt: debug
    custom_components.my_integration: debug
```

**Check logs**:
```bash
tail -f /config/home-assistant.log | grep ERROR
```

### Common Issues

**Entity Unavailable**:
- Check integration connection
- Verify device is powered/online
- Check network connectivity

**Template Errors**:
- Missing `| float` or `| int` filters
- Undefined states (use defaults)
- Syntax errors (missing quotes, braces)

**Database Locks**:
- Too many writes
- Add recorder exclusions
- Increase `commit_interval`

**Slow Automations**:
- Too many `wait_template` (poll every X)
- Use `wait_for_trigger` instead (event-based)
- Simplify templates

---

## 7. Integration Types

### Local Push (Best Performance)

**MQTT**:
```yaml
mqtt:
  sensor:
    - name: "Temperature"
      state_topic: "home/sensor/temperature"
      unit_of_measurement: "°C"
```

**Webhook**:
```yaml
automation:
  - trigger:
      - platform: webhook
        webhook_id: my_webhook
```

### Local Poll (Moderate Performance)

**REST**:
```yaml
sensor:
  - platform: rest
    resource: http://192.168.1.100/api/data
    scan_interval: 60  # Poll every minute
```

### Cloud API (External Dependency)
```yaml
# Example: Weather integration
weather:
  - platform: openweathermap
    api_key: !secret openweather_key
```

---

## 8. Security Best Practices

### Secrets Management
```yaml
# secrets.yaml
api_key: "abc123"
password: "secret"

# configuration.yaml
integration:
  api_key: !secret api_key
```

### Network Security
```yaml
http:
  use_x_forwarded_for: true
  trusted_proxies:
    - 192.168.1.0/24
  ip_ban_enabled: true
  login_attempts_threshold: 5
```

### Automation Safety
```yaml
# Never automate without conditions
- id: safe_automation
  trigger:
    - platform: state
      entity_id: binary_sensor.motion
      to: "on"
  condition:
    # Safety: Don't run at 3am
    - condition: time
      after: "06:00:00"
      before: "23:00:00"
  action:
    - service: light.turn_on
```

---

## 9. Version-Specific Notes

### 2025.11.x Features
- Improved template error messages
- Enhanced automation tracing
- Better entity registry management
- Optimized recorder performance

### Deprecated Features (Avoid)
- Old-style template sensors (use trigger-based)
- `homeassistant.reload_core_config` (use UI)
- Hardcoded entity IDs (use labels/areas)

---

## 10. ClaudeHome Integration Points

### Where ClaudeHome Reads
- `/config/.storage/core.entity_registry` ? Entity list
- `/config/configuration.yaml` ? Integration config
- `/config/automations.yaml` ? Existing automations
- `/config/home-assistant_v2.db` ? Historical data

### Where ClaudeHome Writes
- `/config/automations/claudehome.yaml` ? New automations
- `/config/scripts/claudehome.yaml` ? Helper scripts

### API Calls ClaudeHome Makes
- `POST /api/services/{domain}/{service}` ? Execute actions
- `GET /api/states` ? Fetch current states
- `POST /api/services/automation/reload` ? Reload after changes

---

**Maintained By**: ClaudeHome System  
**Review Schedule**: After major HA version updates  
**Last Reviewed**: 2025-01-31
