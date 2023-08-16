#!/usr/bin/env sh

docker compose build --no-cache
docker compose run wintergatan-data-analysis
