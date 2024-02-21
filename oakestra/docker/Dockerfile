FROM ubuntu:18.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3.6 \
        python3-pip \
        libomp-dev

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install any python packages you need
COPY oakestra/docker/requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

# Install PyTorch and torchvision
#COPY Nvidia_Jetson/setup_pytorch.sh setup_pytorch.sh
#RUN setup_pytorch.sh

# Set the working directory
COPY model/ /app/model
COPY dataset/ /app/dataset
WORKDIR /app

ENV PYTHONPATH="/app"
# Set the entrypoint
ENTRYPOINT [ "python3", "model/train.py" ]
