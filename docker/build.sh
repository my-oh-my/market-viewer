#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Get the current git commit SHA
GIT_COMMIT_SHA=$(git rev-parse --short HEAD)

# Define the image name
IMAGE_NAME="stochastic-oscillator-strategy"

# Build the Docker image with the git commit SHA as a tag
echo "Building Docker image: ${IMAGE_NAME}:${GIT_COMMIT_SHA}"
docker build -t "${IMAGE_NAME}:${GIT_COMMIT_SHA}" -f docker/Dockerfile .

echo "Docker image built and tagged as ${IMAGE_NAME}:${GIT_COMMIT_SHA}"
