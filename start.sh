#!/bin/bash

# LLM Benchmark Platform - Startup Script
#
# Usage:
#   Local:  ./start.sh
#   Docker: docker compose up -d

echo "Starting LLM Benchmark Platform (Streamlit)..."
streamlit run app.py
