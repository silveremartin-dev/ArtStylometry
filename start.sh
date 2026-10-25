#!/usr/bin/env bash
# ArtStylometry - Local Server Launcher (Linux / macOS)

if command -v python3 > /dev/null 2>&1; then
    python3 scripts/serve.py
elif command -v python > /dev/null 2>&1; then
    python scripts/serve.py
else
    echo "Error: Python was not found in PATH."
    exit 1
fi
