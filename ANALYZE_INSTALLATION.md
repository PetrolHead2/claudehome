# Task: Analyze My Home Assistant Installation

I need you to create a comprehensive skill document about MY specific Home Assistant installation.

## Available Data

You have access to:
- Installation analysis: `cat installation_analysis.txt`
- Home Assistant config directory: `/media/pi/NextCloud/homeassistant/`
- Entity registry: `/media/pi/NextCloud/homeassistant/.storage/core.entity_registry`
- Configuration: `/media/pi/NextCloud/homeassistant/configuration.yaml`
- Automations: `/media/pi/NextCloud/homeassistant/automations.yaml`

## What You Should Do

1. Read the installation_analysis.txt file
2. Read the entity registry to understand my entities
3. Read configuration.yaml to see my integrations
4. Read a sample of automations.yaml to understand my patterns
5. Analyze custom components in use

## Output

Create a comprehensive markdown file: `.claude/skills/my_installation.md`

Include:
- **Installation Overview**: Version, type, statistics
- **Entity Breakdown**: Detailed analysis by domain
- **Integration Ecosystem**: What integrations I use and why
- **Automation Patterns**: How I currently automate things
- **Custom Components Analysis**: What each does, why I use it
- **System Characteristics**: Performance, database, resources
- **Privacy Mapping Strategy**: Which entities are sensitive
- **User Profile**: Infer my use cases from the data
- **Optimization Opportunities**: What could be improved
- **ClaudeHome Integration Strategy**: How ClaudeHome should work with MY setup

Be SPECIFIC to my installation - use actual entity counts, integration names, etc.
