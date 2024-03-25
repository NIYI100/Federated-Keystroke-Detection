FROM python:3.9.19-slim-bullseye
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get upgrade -y


# Upgrade pip
RUN pip3 install --upgrade pip

# Install any python packages you need for the model
COPY oakestra/master/docker/requirements.txt master_requirements.txt
RUN pip3 install -r master_requirements.txt

# Start the backend
CMD [ "/run.sh" ]
