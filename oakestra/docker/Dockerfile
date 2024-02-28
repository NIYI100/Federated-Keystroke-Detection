FROM httpd:bookworm

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-full \
        python3.6 \
        python3-pip \
        libomp-dev

# Handle python packages globally
RUN rm /usr/lib/python3.11/EXTERNALLY-MANAGED

# Upgrade pip
RUN pip3 install --upgrade pip

# Install any python packages you need for the model
COPY oakestra/docker/model_requirements.txt model_requirements.txt
RUN pip3 install -r model_requirements.txt

# Install requirements for the backend
COPY oakestra/docker/backend_requirements.txt backend_requirements.txt
RUN pip3 install -r backend_requirements.txt

# Install PyTorch and torchvision
#COPY Nvidia_Jetson/setup_pytorch.sh setup_pytorch.sh
#RUN setup_pytorch.sh

# frontend application
COPY web_app/frontend/ /usr/local/apache2/htdocs/

RUN mkdir /backend
COPY web_app/backend/ /backend

# Set the working directory
COPY model/ /app/model
COPY dataset/ /app/dataset

ENV PYTHONPATH="/app"

COPY oakestra/docker/run.sh /run.sh
RUN chmod +x /run.sh

# Start the backend
CMD [ "/run.sh" ]
