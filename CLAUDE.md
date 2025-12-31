# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ClaudeHome is a Python-based intelligent system administration interface that bridges Home Assistant with Claude AI. It provides autonomous smart home intelligence, analysis, and automation capabilities.

**Tech Stack**: Python 3.11, Anthropic Claude API, Bash scripts
**Target Integration**: Home Assistant 2025.x

## Key Commands

```bash
# Install/update Claude Code binary
./install.sh                    # Install latest stable
./install.sh stable|latest|VERSION  # Install specific version

# Analyze Home Assistant installation (requires HA_CONFIG_DIR env var)
./analyze_installation.sh

# Query Claude with context
./ask-claude.sh 'your question'

# Direct Python API call
./claude_client.py 'task description'
```

## Environment Setup

- **Virtual environment**: `/opt/claudehome/venv` (Python 3.11)
- **Key environment variables** (stored in `.env`):
  - `ANTHROPIC_API_KEY`: API key for Claude
  - `HA_CONFIG_DIR`: Path to Home Assistant config directory

To activate the environment:
```bash
source /opt/claudehome/venv/bin/activate
```

## Architecture

```
/opt/claudehome/
├── .claude/skills/         # AI knowledge base (Claude context)
│   ├── claude_capabilities.md  # Model capabilities & best practices
│   └── home_assistant_core.md  # HA reference & automation syntax
├── venv/                   # Python virtual environment
├── claude_client.py        # Python client for Anthropic API
├── install.sh              # Claude Code installer
├── analyze_installation.sh # HA installation analyzer
└── ask-claude.sh           # Helper script for Claude queries
```

### Skills System

The `.claude/skills/` directory contains knowledge base files that provide context:
- `claude_capabilities.md`: Model limitations, prompt engineering, cost optimization
- `home_assistant_core.md`: YAML automation syntax, templates, common patterns

### API Integration

`claude_client.py` uses `claude-sonnet-4-20250514` model with 8192 max tokens. The client is minimal - it sends a user message and returns the response text.

## Home Assistant Integration Points

ClaudeHome reads from:
- `/config/.storage/core.entity_registry` - Entity list
- `/config/configuration.yaml` - Integration config
- `/config/automations.yaml` - Existing automations
- `/config/home-assistant_v2.db` - Historical data (SQLite)

ClaudeHome writes to:
- `/config/automations/claudehome.yaml` - New automations
- `/config/scripts/claudehome.yaml` - Helper scripts

## Security Guidelines

When generating code or automations:
- Never hardcode secrets (use `!secret` references)
- Always include conditions in automations (prevent unintended triggers)
- Validate inputs and include error handling
- Suggest backups before major configuration changes
- Use least privilege principle for API access
