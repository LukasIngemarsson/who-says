#!/bin/bash

COMMAND=${1:-""}
IMAGE=test
IMAGE_PATH=images/$IMAGE
IMAGE_NAME="$IMAGE:$USER"
CONTAINER_NAME="dev-$USER-$IMAGE"

if [[ "$COMMAND" = "help" || "$COMMAND" = "-h" || "$COMMAND" = "--help" || "$COMMAND" = "" ]]; then
    echo "Usage: $0 COMMAND [OPTION...]"
    echo " Commands:"     
    echo "   build          - Build the container."
    echo "   start          - Start the container (builds it if it has changed)."
    echo "   bash           - Access the running container from the terminal."
    exit 0
fi

if [[ "$COMMAND" = "build" ]]; then
    docker build $IMAGE_PATH -t $IMAGE_NAME
    exit 0
fi

if [[ "$COMMAND" = "start" ]]; then
    docker run -it \
      --rm \
      --gpus all \
      --name $CONTAINER_NAME \
      --volume=/tmp/.X11-unix/:/tmp/.X11-unix/\
      --volume=$(pwd):/home/$USER \
      --workdir="/home/$USER" \
      --env HOME="/home/$USER" \
      --env DISPLAY=$DISPLAY \
      $IMAGE_NAME
    exit 0
fi

if [[ "$COMMAND" = "bash" ]]; then
    docker exec -it $CONTAINER_NAME /bin/bash
    exit 0