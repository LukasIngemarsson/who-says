# #!/bin/bash

# docker build -t whosays-app .

# # Create embeddings directory if it doesn't exist
# mkdir -p embeddings

# CONTAINER_NAME="whosays-container"

# # Stop and remove existing container if it exists
# docker stop $CONTAINER_NAME 2>/dev/null
# docker rm $CONTAINER_NAME 2>/dev/null

# # Create and start new container
# if [ -f .env ]; then
#   docker run -d --name $CONTAINER_NAME -p 8000:8000 \
#     -v "$(pwd)/.env:/app/.env" \
#     -v "$(pwd)/embeddings:/app/embeddings" \
#     whosays-app
# else
#   docker run -d --name $CONTAINER_NAME -p 8000:8000 \
#     -v "$(pwd)/embeddings:/app/embeddings" \
#     whosays-app
# fi

# echo "Container $CONTAINER_NAME is running. View logs with: docker logs -f $CONTAINER_NAME"


#!/bin/bash
set -e

IMAGE_NAME="whosays-app"
CONTAINER_NAME="whosays-container"

# Build image
docker build -t "$IMAGE_NAME" .

# Create embeddings directory if it doesn't exist
mkdir -p embeddings

# Stop and remove existing container if it exists
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

# Detect if Docker supports GPUs (NVIDIA toolkit installed)
GPU_ARGS=""
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi "nvidia"; then
  # Old style (nvidia runtime present) – --gpus should still work
  GPU_ARGS="--gpus all"
elif docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  # Newer toolkit, --gpus works
  GPU_ARGS="--gpus all"
fi

echo "Using GPU args: '$GPU_ARGS'"

# Create and start new container
if [ -f .env ]; then
  docker run -d --name "$CONTAINER_NAME" -p 8000:8000 \
    $GPU_ARGS \
    -v "$(pwd)/.env:/app/.env" \
    -v "$(pwd)/embeddings:/app/embeddings" \
    "$IMAGE_NAME"
else
  docker run -d --name "$CONTAINER_NAME" -p 8000:8000 \
    $GPU_ARGS \
    -v "$(pwd)/embeddings:/app/embeddings" \
    "$IMAGE_NAME"
fi

echo "Container $CONTAINER_NAME is running. View logs with: docker logs -f $CONTAINER_NAME"
