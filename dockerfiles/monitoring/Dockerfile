FROM python:3.11-slim

# 1) Arbeitsverzeichnis im Container
WORKDIR /app

# 2) System-Abhängigkeiten installieren: Supervisor, Git
RUN apt-get update && apt-get install -y --no-install-recommends \
      supervisor \
      git \
    && rm -rf /var/lib/apt/lists/*

# 3) Kopiere requirements und installiere Python-Abhängigkeiten + DVC
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir dvc

# 4) Kopiere den Drift-Monitor-Code ins Image
COPY drift_monitor /app/drift_monitor

# 5) Supervisor-Konfiguration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 6) Volumes definieren, exakt so, wie dein Skript die Pfade erwartet
VOLUME ["/shared_volume/data/raw", "/shared_volume/data/feedback", "/shared_volume/data/processed"]

# 7) Supervisor im Vordergrund starten
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
