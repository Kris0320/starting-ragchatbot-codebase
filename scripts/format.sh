#!/bin/bash
set -e

echo "Formatting code..."

echo ""
echo "--- black ---"
uv run black main.py backend/

echo ""
echo "Formatting complete!"
