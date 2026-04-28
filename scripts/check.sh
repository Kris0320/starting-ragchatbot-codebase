#!/bin/bash
set -e

echo "Running code quality checks..."

echo ""
echo "--- black (format check) ---"
uv run black --check main.py backend/

echo ""
echo "All checks passed!"
