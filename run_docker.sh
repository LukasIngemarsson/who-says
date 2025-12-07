#!/bin/bash

docker build -t whosays-app .

# Create embeddings directory if it doesn't exist
mkdir -p embeddings

CONTAINER_NAME="whosays-container"

# Stop and remove existing container if it exists
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Create and start new container
if [ -f .env ]; then
  docker run -d --name $CONTAINER_NAME -p 8000:8000 \
    -v "$(pwd)/.env:/app/.env" \
    -v "$(pwd)/embeddings:/app/embeddings" \
    whosays-app
else
  docker run -d --name $CONTAINER_NAME -p 8000:8000 \
    -v "$(pwd)/embeddings:/app/embeddings" \
    whosays-app
fi

echo "Container $CONTAINER_NAME is running. View logs with: docker logs -f $CONTAINER_NAME"