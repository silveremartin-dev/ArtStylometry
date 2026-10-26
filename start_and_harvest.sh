#!/usr/bin/env bash
# ArtStylometry - Local Server & Background Database Harvester (Linux / macOS)

if command -v python3 > /dev/null 2>&1; then
    python3 scripts/serve_and_harvest.py "$@"
elif command -v python > /dev/null 2>&1; then
    python scripts/serve_and_harvest.py "$@"
else
    echo "Error: Python was not found in PATH."
    exit 1
fi
