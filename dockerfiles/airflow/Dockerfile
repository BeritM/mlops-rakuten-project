FROM apache/airflow:2.8.0-python3.11

USER root

# System Updates und Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    docker.io \
    docker-compose \
    git \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Fix Docker permissions
# Create docker group if it doesn't exist and add airflow user
RUN groupadd -f -g 999 docker || true && \
    usermod -aG docker airflow && \
    # Give airflow user sudo access to docker commands (alternative approach)
    echo "airflow ALL=(ALL) NOPASSWD: /usr/bin/docker, /usr/bin/docker-compose, /usr/local/bin/docker-compose" >> /etc/sudoers

# Ensure correct permissions on docker socket (will be mounted from host)
RUN touch /var/run/docker.sock && \
    chmod 666 /var/run/docker.sock || true

# Git Konfiguration für Container
RUN git config --system --add safe.directory '*'

# Docker Compose V2 installieren (neueste Version)
RUN curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose && \
    chmod +x /usr/local/bin/docker-compose

# Create necessary directories with correct permissions
RUN mkdir -p /opt/airflow/shared_volume /opt/airflow/project && \
    chown -R airflow:root /opt/airflow/shared_volume /opt/airflow/project && \
    chmod -R 775 /opt/airflow/shared_volume /opt/airflow/project

USER airflow

# Python Dependencies für Airflow
RUN pip install --no-cache-dir \
    apache-airflow-providers-docker==3.8.0 \
    python-dotenv \
    docker

# DVC installieren für den Airflow Container (falls direkte DVC Operationen nötig sind)
RUN pip install --no-cache-dir "dvc[s3]"

# Set proper Python path
ENV PYTHONPATH=/opt/airflow/project:$PYTHONPATH