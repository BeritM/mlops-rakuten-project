FROM apache/airflow:2.8.0-python3.11

USER root

# Install Docker CLI for Docker-in-Docker
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Add Airflow user to Docker Group
RUN groupadd -f docker && \
    usermod -aG docker airflow

USER airflow

# Docker Provider
RUN pip install apache-airflow-providers-docker==3.8.0

# Copy requirements if you have any additional ones
# COPY requirements-airflow.txt /
# RUN pip install -r /requirements-airflow.txt