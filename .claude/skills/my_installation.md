# My Home Assistant Installation

**Purpose**: Specific knowledge about THIS Home Assistant installation
**Location**: Dianavägen 15, Täby, Stockholm, Sweden (SE3 electricity zone)
**Last Updated**: 2025-12-31

---

## 1. Installation Overview

### System Stats
| Metric | Value |
|--------|-------|
| HA Version | 2025.11.3 |
| Total Entities | 1,729 |
| Automations | 63 |
| Custom Components | 18 |
| Database | External MariaDB (172.21.10.130:3306) |
| Config Storage | 4.2 GB (4.0 GB backups) |
| Host Memory | 15 GB (3.8 GB available) |
| Docker Containers | 15 running |

### Deployment Architecture
```
Host: Debian Linux (debian.koffern.hopto.org)
├── Docker: homeassistant/home-assistant:stable
├── Docker: mariadbHomeAssistant (MariaDB)
├── InfluxDB: 192.168.50.250:8086
└── Config: /media/pi/NextCloud/homeassistant/
```

### External Access
- Domain: koffern.duckdns.org
- SSL: Let's Encrypt via dehydrated
- Google Assistant integration enabled

---

## 2. Entity Breakdown

### By Domain (Top 15)
| Domain | Count | Primary Use |
|--------|-------|-------------|
| sensor | 1,018 | Energy, weather, car, prices |
| binary_sensor | 135 | Motion, occupancy, states |
| switch | 129 | Power outlets, Frigate controls |
| device_tracker | 97 | Family phones, car |
| automation | 63 | See automation patterns below |
| image | 42 | Camera snapshots |
| number | 36 | Settings, thresholds |
| zone | 30 | Location tracking |
| update | 28 | Component updates |
| light | 20 | IKEA Dirigera, Tuya |
| select | 20 | Options/modes |
| button | 19 | Actions (charging, etc.) |
| media_player | 13 | Sonos, displays |
| input_number | 12 | User settings |
| camera | 11 | Frigate NVR cameras |

### Key Entity Categories

**Energy Monitoring** (~300+ entities):
- `sensor.tibber_pulse_dianavagen_15_*` - Main electricity meter (Tibber)
- `sensor.nordpool_kwh_se3_sek_3_10_025` - Electricity spot prices
- `sensor.stickpropp_*` - Individual outlet power monitoring (10-18)
- `sensor.*_energy`, `sensor.*_power` - Powercalc calculated values
- `sensor.totalpriceperkwh` - Total electricity cost including fees

**Car (Mercedes JBB78W - Plug-in Hybrid)**:
- `sensor.jbb78w_*` - ~50+ entities via mbapi2020
- `sensor.jbb78w_state_of_charge` - Battery percentage
- `sensor.jbb78w_range_electric` - Electric range
- `sensor.jbb78w_fuel_level` - Petrol tank level
- `sensor.jbb78w_odometer` - Total distance
- `switch.jbb78w_pre_entry_climate_control` - Climate pre-conditioning

**EV Charger (Zaptec "laddbox")**:
- `sensor.laddbox_charger_mode` - Charging state
- `sensor.laddbox_charge_power` - Current charging power
- `sensor.laddbox_session_total_charge` - Session energy
- `button.laddbox_authorize_charging` - Start charging
- `button.laddbox_stop_charging` - Stop charging
- `button.laddbox_resume_charging` - Resume charging

**Security (Frigate NVR)**:
- Cameras: frigate_15, frigate_17, frigate_147, frigate_157, frigate_187, frigate_34, frigate_44
- Cameras: besder_1, besder_2, besder_3, besder_4, besder_5
- `switch.frigate_*_detect` - Detection toggles
- `switch.frigate_*_motion` - Motion detection toggles
- `switch.frigate_*_recordings` - Recording toggles
- `binary_sensor.*_person_occupancy` - Person detection

**Family Tracking (OwnTracks via MQTT)**:
- `device_tracker.magnus_mobil` - Owner
- `input_text.magnusregion` - Magnus current region
- `input_text.victoriaregion` - Victoria current region
- `input_text.williamregion` - William current region
- `input_text.asiaregion` - Asia current region
- `binary_sensor.template_binary_someone_home` - Anyone home

**Fuel Prices (REST sensors)**:
- `sensor.st1_ica_stop_95` - Primary petrol station
- Multiple stations: Ingo, Preem, Tanka in Täby area

---

## 3. Custom Components Analysis

| Component | Purpose | Key Entities |
|-----------|---------|--------------|
| **mbapi2020** | Mercedes-Benz integration | JBB78W car data, charging status |
| **zaptec** | EV charger control | laddbox_* entities |
| **nordpool** | Electricity spot prices | nordpool_kwh_se3_sek_* |
| **frigate** | NVR/AI camera detection | 11 cameras, motion/person detection |
| **powercalc** | Virtual power sensors | Calculated energy consumption |
| **energy_meter** | Cost tracking per device | Daily/monthly cost calculations |
| **dirigera_platform** | IKEA smart home hub | Lights, switches via Dirigera |
| **hacs** | Custom component store | Component management |
| **landroid_cloud** | Worx lawn mower | Mowing schedule, status |
| **sector** | Sector alarm system | Home security |
| **burze_dzis_net** | Storm/weather warnings | Polish weather service |
| **bensinpriser** | Fuel price tracking | Swedish petrol prices |
| **webrtc** | Camera streaming | Low-latency video |
| **roborock_custom_map** | Robot vacuum maps | Vacuum zone control |
| **xiaomi_cloud_map_extractor** | Vacuum maps | Room mapping |
| **perific-meter** | Energy meter (Enegic) | Grid connection monitoring |
| **bluetooth_tracker** | BT presence detection | Device tracking |
| **PR** | Unknown/custom | Needs investigation |

---

## 4. Automation Patterns

### Primary Automation Categories

**1. EV Charging Optimization (20+ automations)**
Most complex automation area. Key patterns:
- Charge during cheapest Nordpool hours (22:00-06:00)
- Compare electricity cost vs petrol cost per km
- Stop charging if hourly consumption exceeds 5 kWh
- Auto-authorize when cable plugged in (if Magnus home)
- Daytime charging limited to 3 kWh max
- Calculate optimal charging start time using `cheapest_energy_hours.jinja`

Key automations:
- `JBB Start Charging Cheapest, Night time` - Uses calculated cheapest hours
- `JBB Disable Charging 06-02` - Prevent charging during peak hours
- `JBB Disable Charging when Electricity is higher then Petrol` - Cost comparison
- `JBB Auto Authorization When Cable Plugged in` - Convenience automation
- `Electrical Extreme High Consumption` - Emergency stop at 9 kWh/hour

**2. Security/Surveillance (15+ automations)**
- Frigate detection ON when Magnus leaves home
- Frigate detection OFF when Magnus arrives home
- Camera notifications via blueprint (SgtBatten/Beta.yaml)
- Kitchen display shows cameras on motion detection
- Motion on property triggers surveillance view

**3. Energy Monitoring (5+ automations)**
- High consumption warnings (>5 kWh/hour)
- Extreme consumption emergency stops
- Price indicator light (RGB based on price level)
- Hourly consumption tracking
- Lost connection alerts (Enegic meter)

**4. Family Tracking (4 automations)**
- OwnTracks MQTT → input_text region updates
- Per-person: Magnus, Victoria, William, Asia

**5. UI/Display Control**
- Kitchen display shows Sonos controls on music
- Electricity price shown on display hourly
- Camera feeds on motion

**6. Maintenance**
- Weekly backup (Monday 4am)
- Nordpool reload at 21:00 if tomorrow prices missing
- Error log notifications via Discord

### Automation Style Observations
- Uses timestamp-based IDs (e.g., `'1725306210562'`)
- Heavy use of template conditions
- Blueprints for Frigate notifications
- `mode: single` is default
- Swedish/English mix in naming
- Complex Jinja2 templates for charging logic

---

## 5. Energy Management Strategy

### Electricity Pricing
- **Provider**: Nordpool SE3 zone
- **Meter**: Tibber Pulse (real-time consumption)
- **Price sensor**: `sensor.nordpool_kwh_se3_sek_3_10_025`
- **Total price**: `sensor.totalpriceperkwh` (includes fees)
- **Price indicator**: RGB light shows red/yellow/green based on price

### Monitored Circuits (via Stickpropp outlets 10-18)
Each outlet tracks: power, energy, voltage, current
- Kitchen appliances
- Living room (vardagsrum)
- Outdoor outlets (1-6)
- Dishwasher
- Background consumption
- VVB (hot water heater) standby
- Fridge/freezer

### Cost Calculation
Uses YAML anchors for price entities:
```yaml
price_entity: &entity-price sensor.nordpool_kwh_se3_sek_3_10_025
price_entity: &entity-total-price sensor.totalpriceperkwh
```

Daily and monthly cost tracking per circuit via `energy_meter` integration.

### Car Charging Economics
Compares:
- `sensor.price_per_mil` - Electricity cost per 10km
- `number.jbb78w_fuel_consumption` - Petrol cost per 10km
- Stops charging if electricity more expensive than petrol

---

## 6. Database & Recording Strategy

### MariaDB Configuration
- External database: `mysql://root:***@172.21.10.130:3306/homeassistant`
- Retention: 7000 days (effectively forever)

### Excluded from Recording (High-frequency)
```yaml
entity_globs:
  - sensor.*_power
  - sensor.openweathermap_*
  - sensor.stocksund_*
  - sensor.tibber_pulse_dianavagen_15_voltage_*
  - sensor.tibber_pulse_dianavagen_15_current_*
  - sensor.stickpropp_*_consumed_last_updated
  - sensor.stickpropp_*_current_voltage
  - sensor.stickpropp_*_amps
entities:
  - sun.sun
  - sensor.frigate_* (performance metrics)
  - sensor.mariadb_uptime/questions
```

### InfluxDB (Long-term Analytics)
- Server: 192.168.50.250:8086
- Bucket: homeassistant
- Includes: sensor, binary_sensor, device_tracker, number, input_number, person, select, switch, input_datetime, input_text
- Same exclusions as recorder

---

## 7. Sensitive/Private Entities

### Location Data
- All device_tracker entities
- Zone definitions (30 zones)
- OwnTracks MQTT topics
- input_text.*region entities

### Personal Identifiers
- person.* entities (4 family members)
- Phone device trackers

### Security
- Alarm: sector integration
- Camera feeds and recordings
- Frigate detection states

### Financial
- Energy costs and consumption
- Car fuel costs
- Electricity prices

### Credentials (in config - DO NOT EXPOSE)
- `.env` file contains API keys
- `secrets.yaml` for passwords
- MariaDB credentials in config
- InfluxDB token in config
- SMTP password for notifications
- Google service account

---

## 8. User Profile (Inferred)

### Household
- **Owner**: Magnus (primary phone: magnus_mobil)
- **Family**: 4 persons (Magnus, Victoria, William, Asia)
- **Location**: Dianavägen 15, Täby (suburb of Stockholm)
- **Property**: House with garden (has lawn mower, multiple outdoor outlets)

### Vehicles
- Mercedes plug-in hybrid (JBB78W)
- Zaptec home charger
- Heavy optimization of charging costs

### Interests/Priorities
1. **Energy cost optimization** - Primary focus
2. **Home security** - 11+ cameras with AI detection
3. **Automation** - 63 automations, mostly energy-related
4. **Smart home** - IKEA, Tuya, Sonos ecosystem

### Technical Level
- Advanced user (custom components, complex templates)
- Uses YAML anchors, Jinja2 macros
- External databases, InfluxDB analytics
- Docker-based deployment

---

## 9. Optimization Opportunities

### Potential Improvements

**1. Automation Naming**
Current IDs are timestamps. Consider descriptive IDs:
```yaml
# Current: '1725306210562'
# Better: backup_monday_4am
```

**2. Template Consolidation**
Multiple similar power calculation templates could use a macro.

**3. Recorder Optimization**
Consider excluding more high-frequency sensors to reduce database growth.

**4. Blueprint Adoption**
Frigate notifications use blueprints - could extend to other patterns.

**5. Package Organization**
Consider splitting configuration into packages:
```
packages/
  car_charging/
  energy_monitoring/
  security/
```

---

## 10. ClaudeHome Integration Strategy

### High-Value Use Cases

**1. Charging Optimization Analysis**
- Analyze charging patterns vs prices
- Suggest optimal charging windows
- Compare actual vs predicted costs

**2. Anomaly Detection**
- Unusual energy consumption patterns
- Device failures (power drops to 0)
- Security events correlation

**3. Automation Suggestions**
- Based on observed patterns
- Energy-saving opportunities
- Presence-based optimizations

**4. Troubleshooting**
- Parse error logs (`sensor.hafail`)
- Identify entity unavailability
- Database performance

### API Integration Points

**Read from**:
- Entity registry for current entities
- State history for patterns
- Automations for current logic

**Could write to** (with approval):
- New automations in `/config/automations/claudehome.yaml`
- Helper scripts
- Input number/boolean adjustments

### Context for Prompts
When analyzing this installation, always consider:
- Swedish context (SE3 electricity zone, SEK currency)
- Family of 4 with varying presence patterns
- Plug-in hybrid vehicle with complex charging logic
- Heavy focus on cost optimization
- 11 security cameras that toggle based on presence

---

**Maintained By**: ClaudeHome System
**Review Schedule**: After significant installation changes
**Last Reviewed**: 2025-12-31
