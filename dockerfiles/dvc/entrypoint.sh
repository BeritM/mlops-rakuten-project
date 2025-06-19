#!/bin/sh
set -e

CONTAINER_UID=${AIRFLOW_UID:-50000}
CONTAINER_GID=${AIRFLOW_GID:-0}


# Überprüfen, ob das Verzeichnis /app/shared_volume leer ist
# Wenn ja, initialisiere die Berechtigungen
if [ -z "$(ls -A /app/shared_volume)" ]; then
    echo "Initializing /app/shared_volume permissions for user ${CONTAINER_UID}:${CONTAINER_GID}"
    # Erstelle die nötigen Unterverzeichnisse (falls sie noch nicht existieren)
    mkdir -p /app/shared_volume/data/feedback /app/shared_volume/data/processed /app/shared_volume/data/raw /app/shared_volume/models
    # Setze den Besitz für das gesamte Volume auf den gewünschten Benutzer
    chown -R ${CONTAINER_UID}:${CONTAINER_GID} /app/shared_volume
fi

# Führe den ursprünglichen CMD-Befehl aus
exec "$@"