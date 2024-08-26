#!/bin/bash
# Build Docker image using the provided Dockerfile

# Load Docker module if needed
module load docker

# Run Docker build command
docker build -f ./lsv_cluster_files/mamba.dockerfile --build-arg USER_UID=$UID --build-arg USER_NAME=$(id -un) -t docker.lsv.uni-saarland.de/hhwang/mamba:0 .

