#!/bin/bash
# Set up the environment and run the discovery pipeline.
uv sync                              # install dependencies into .venv
cp -n config.example.yaml config.yaml || true   # create config on first run
uv run discover.py                   # full pipeline, 25 repos, all formats
