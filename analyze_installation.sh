#!/bin/bash

echo "=== Home Assistant Installation Analysis ==="
echo ""
echo "Analysis Date: $(date)"
echo "Analysis Host: $(hostname)"
echo ""

echo "## Installation Paths"
echo "- Config directory: $HA_CONFIG_DIR"
echo "- ClaudeHome directory: /opt/claudehome"
echo ""

echo "## Home Assistant Version"
if [ -f "$HA_CONFIG_DIR/.HA_VERSION" ]; then
    HA_VERSION=$(cat $HA_CONFIG_DIR/.HA_VERSION)
    echo "- Version: $HA_VERSION"
else
    echo "- Version: Unknown (file not found)"
fi
echo ""

echo "## Database Information"
if [ -f "$HA_CONFIG_DIR/home-assistant_v2.db" ]; then
    DB_SIZE=$(ls -lh $HA_CONFIG_DIR/home-assistant_v2.db | awk '{print $5}')
    DB_MODIFIED=$(stat -c '%y' $HA_CONFIG_DIR/home-assistant_v2.db | cut -d'.' -f1)
    echo "- Database size: $DB_SIZE"
    echo "- Last modified: $DB_MODIFIED"
    
    # Count records (if sqlite3 is available)
    if command -v sqlite3 &> /dev/null; then
        STATE_COUNT=$(sqlite3 $HA_CONFIG_DIR/home-assistant_v2.db "SELECT COUNT(*) FROM states;" 2>/dev/null || echo "N/A")
        EVENT_COUNT=$(sqlite3 $HA_CONFIG_DIR/home-assistant_v2.db "SELECT COUNT(*) FROM events;" 2>/dev/null || echo "N/A")
        echo "- State records: $STATE_COUNT"
        echo "- Event records: $EVENT_COUNT"
    fi
else
    echo "- Database: Not found"
fi
echo ""

echo "## Entity Registry"
if [ -f "$HA_CONFIG_DIR/.storage/core.entity_registry" ]; then
    REGISTRY_SIZE=$(ls -lh $HA_CONFIG_DIR/.storage/core.entity_registry | awk '{print $5}')
    echo "- Registry size: $REGISTRY_SIZE"
    
    if command -v jq &> /dev/null; then
        ENTITY_COUNT=$(jq '.data.entities | length' $HA_CONFIG_DIR/.storage/core.entity_registry 2>/dev/null)
        echo "- Total entities: $ENTITY_COUNT"
        echo ""
        echo "### Entity Breakdown by Domain:"
        jq -r '.data.entities[].entity_id' $HA_CONFIG_DIR/.storage/core.entity_registry 2>/dev/null | \
            cut -d'.' -f1 | sort | uniq -c | sort -rn
    fi
else
    echo "- Entity registry: Not found"
fi
echo ""

echo "## Automations"
if [ -f "$HA_CONFIG_DIR/automations.yaml" ]; then
    AUTO_COUNT=$(grep -c '^- id:' $HA_CONFIG_DIR/automations.yaml 2>/dev/null || echo "0")
    AUTO_SIZE=$(ls -lh $HA_CONFIG_DIR/automations.yaml | awk '{print $5}')
    echo "- Count: $AUTO_COUNT"
    echo "- File size: $AUTO_SIZE"
    echo ""
    echo "### Sample Automation IDs (first 10):"
    grep '^- id:' $HA_CONFIG_DIR/automations.yaml | head -10 | sed 's/^- id: /  - /'
else
    echo "- Automations file: Not found"
fi
echo ""

echo "## Scripts"
if [ -f "$HA_CONFIG_DIR/scripts.yaml" ]; then
    SCRIPT_SIZE=$(ls -lh $HA_CONFIG_DIR/scripts.yaml | awk '{print $5}')
    echo "- Scripts file size: $SCRIPT_SIZE"
fi
echo ""

echo "## Configuration"
if [ -f "$HA_CONFIG_DIR/configuration.yaml" ]; then
    CONFIG_SIZE=$(ls -lh $HA_CONFIG_DIR/configuration.yaml | awk '{print $5}')
    echo "- Configuration size: $CONFIG_SIZE"
    echo ""
    echo "### Top-level integrations (first 25):"
    grep -E '^[a-z_]+:' $HA_CONFIG_DIR/configuration.yaml | head -25 | sed 's/^/  - /'
fi
echo ""

echo "## Custom Components"
if [ -d "$HA_CONFIG_DIR/custom_components" ]; then
    CUSTOM_COUNT=$(ls -1 $HA_CONFIG_DIR/custom_components | wc -l)
    echo "- Count: $CUSTOM_COUNT"
    echo "- Components:"
    ls -1 $HA_CONFIG_DIR/custom_components | sed 's/^/  - /'
else
    echo "- Custom components: None"
fi
echo ""

echo "## Storage Usage"
echo "- Total HA config directory: $(du -sh $HA_CONFIG_DIR | awk '{print $1}')"
echo "- Breakdown:"
du -sh $HA_CONFIG_DIR/* 2>/dev/null | sort -rh | head -10 | sed 's/^/  /'
echo ""

echo "## System Resources (Host)"
echo "- Available disk: $(df -h / | tail -1 | awk '{print $4}')"
echo "- Total memory: $(free -h | grep Mem | awk '{print $2}')"
echo "- Available memory: $(free -h | grep Mem | awk '{print $7}')"
echo ""

echo "## Docker Status"
if command -v docker &> /dev/null; then
    echo "- Docker version: $(docker --version)"
    echo "- Running containers: $(docker ps | grep -v CONTAINER | wc -l)"
    echo ""
    echo "### HA Related Containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" | grep -i home || echo "  None found"
fi

echo ""
echo "=== Analysis Complete ==="
