#!/usr/bin/env python3
import anthropic
import os
import sys

def execute_task(task_description):
    """Execute a task using Claude API"""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{
            "role": "user",
            "content": task_description
        }]
    )
    
    return message.content[0].text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./claude_client.py 'task description'")
        sys.exit(1)
    
    task = " ".join(sys.argv[1:])
    result = execute_task(task)
    print(result)
