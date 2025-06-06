FROM python:3.11-slim

WORKDIR /app

# 1) System-Paket-Installation: Supervisor und Git (für DVC-Pull)
RUN apt-get update && apt-get install -y --no-install-recommends \
      supervisor \
      git \
    && rm -rf /var/lib/apt/lists/*

# 2) Python-Dependencies installieren
#    (requirements.txt liegt im Ordner dockerfiles/monitoring/)
COPY dockerfiles/monitoring/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir dvc

# 3) Drift-Monitor-Code ins Image kopieren
#    (drift_monitor/ befindet sich im Projekt-Root direkt neben dockerfiles/)
COPY drift_monitor /app/drift_monitor


# 4) Supervisor-Konfiguration ins Image kopieren
#    (supervisord.conf liegt im Ordner dockerfiles/monitoring/)
COPY dockerfiles/monitoring/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 5) Declare volume-Points (gemountet später durch docker-compose)
#    Hier reservieren wir /app/shared_volume (für DVC-Daten) und /app/drift_monitor (für Reports).
VOLUME ["/app/shared_volume", "/app/drift_monitor"]

# 6) Beim Container-Start Supervisor im Vordergrund starten
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]