FROM python:3.11-slim

WORKDIR /app

# 1) System-Dependencies für DVC
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# 2) Python-Packages installieren
#    requirements.txt liegt in dockerfiles/monitoring/
COPY dockerfiles/monitoring/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir dvc[s3]

# 3) App-Code kopieren
#    monitor.py liegt in monitor/monitor.py
COPY monitor/monitor.py ./monitor.py

# 4) Entrypoint kopieren & ausführbar machen
#    entrypoint.sh liegt in dockerfiles/monitoring/
COPY dockerfiles/monitoring/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

# 5) DVC-Metadaten kopieren
COPY .dvc/ .dvc/
COPY *.dvc ./

# 6) Startbefehl
ENTRYPOINT ["./entrypoint.sh"]