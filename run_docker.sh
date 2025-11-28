#!/bin/bash

docker build --no-cache -t whosays-app .
docker run -p 8000:8000 -v $(pwd)/.env:/app/.env whosays-app