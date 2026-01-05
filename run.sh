#!/bin/bash

COMMAND=${1:-""}
IMAGE=test
IMAGE_PATH=images/$IMAGE
# Use USERNAME on Windows, USER on Linux/Mac
CURRENT_USER=${USER:-$USERNAME}
IMAGE_NAME="$IMAGE:$CURRENT_USER"
CONTAINER_NAME="dev-$CURRENT_USER-$IMAGE"

if [[ "$COMMAND" = "help" || "$COMMAND" = "-h" || "$COMMAND" = "--help" || "$COMMAND" = "" ]]; then
    echo "Usage: $0 COMMAND [OPTION...]"
    echo " Commands:"
    echo "   build            - Build the container."
    echo "   start            - Start the container (builds it if it has changed)."
    echo "   bash             - Access the running container from the terminal."
    echo "   rebuild-frontend - Rebuild the frontend inside the running container."
    exit 0
fi

if [[ "$COMMAND" = "build" ]]; then
    docker build -f $IMAGE_PATH/Dockerfile -t $IMAGE_NAME .
    exit 0
fi

if [[ "$COMMAND" = "start" ]]; then
    MSYS_NO_PATHCONV=1 docker run -it \
      --rm \
      --gpus all \
      --name $CONTAINER_NAME \
      --volume=/tmp/.X11-unix/:/tmp/.X11-unix/ \
      --volume="$(pwd):/home/$CURRENT_USER" \
      --workdir="/home/$CURRENT_USER" \
      --env HOME="/home/$CURRENT_USER" \
      --env DISPLAY=$DISPLAY \
      -p 5000:5000 \
      $IMAGE_NAME
    exit 0
fi

if [[ "$COMMAND" = "bash" ]]; then
    docker exec -it $CONTAINER_NAME /bin/bash
    exit 0
fi

if [[ "$COMMAND" = "rebuild-frontend" ]]; then
    echo "Rebuilding frontend inside container..."
    docker exec -it $CONTAINER_NAME /bin/bash -c "cd frontend && npm install && npm run build && cp -r dist/* /app/client/"
    echo "Frontend rebuilt successfully!"
    exit 0
fi