#!/usr/bin/env bash
# GeoAI Lab - Episode 1: run OpenDroneMap on a folder of drone photos (Docker, Linux/macOS/WSL)
# Layout expected:   $DATA/aukerman/images/*.JPG
# Outputs land in:   $DATA/aukerman/odm_orthophoto, odm_dem, odm_georeferencing, odm_report
DATA="${1:-$(pwd)/drone}"
PROJECT="${2:-aukerman}"
docker run --rm -v "$DATA:/datasets" opendronemap/odm \
    --project-path /datasets "$PROJECT" \
    --dsm --dtm \
    --orthophoto-resolution 5 \
    --dem-resolution 10 \
    --pc-quality medium \
    --feature-quality high \
    --max-concurrency 10
