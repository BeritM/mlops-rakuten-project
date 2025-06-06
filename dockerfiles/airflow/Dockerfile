FROM apache/airflow:2.8.0-python3.11

# ----- Root-Phase ------------------------------------------------------------
USER root

# 1) Voraussetzungen für https-APT
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg

# 2) Offizielles Docker-Repo hinzufügen
RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg \
         | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) \
         signed-by=/etc/apt/keyrings/docker.gpg] \
         https://download.docker.com/linux/debian bookworm stable" \
         > /etc/apt/sources.list.d/docker.list

# 3) Docker Engine CLI + Compose-Plugin installieren
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        docker-ce-cli \
        docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/*

# 4) Airflow-User in Docker-Gruppe
RUN groupadd -f docker && \
    usermod -aG docker airflow

USER airflow

# 5) Optional: Airflow-Docker-Provider
RUN pip install "apache-airflow-providers-docker==3.8.0"
