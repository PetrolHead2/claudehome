#!/bin/bash
# Helper script to ask Claude questions with context

cd /opt/claudehome

# If no argument, show usage
if [ $# -eq 0 ]; then
    echo "Usage: ./ask-claude.sh 'your question'"
    echo "Example: ./ask-claude.sh 'Analyze my HA installation'"
    exit 1
fi

# Run Claude with the question
claude "$@"
