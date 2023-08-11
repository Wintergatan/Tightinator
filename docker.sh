#!/bin/bash

docker buildx build . -t yanfett/wintergatan-data-analysis:latest
docker run -d --network host yanfett/wintergatan-data-analysis:latest