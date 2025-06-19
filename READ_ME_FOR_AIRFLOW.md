# Airflow Setup - Plattformunabhängig

## Voraussetzungen
- Docker & Docker Compose installiert
- Port 8080 frei

## Setup (einmalig)

```bash
# 1. Ordner erstellen
mkdir -p dags logs dockerfiles/airflow

# 2. Airflow initialisieren
docker-compose up airflow-init # Only has to be executed once

# 3. Alle Services starten
docker-compose up -d
```

## Verwendung

### Services starten/stoppen
```bash
# Starten
docker-compose up -d

# Stoppen
docker-compose down

# Logs anschauen
docker-compose logs -f airflow-scheduler
```

### Airflow UI
- URL: http://localhost:8080
- Login: airflow / airflow

### ML Pipeline manuell starten (ohne Airflow)
```bash
# Komplette Pipeline
docker-compose run --rm dvc-sync
docker-compose run --rm preprocessing
docker-compose run --rm model_training
docker-compose run --rm model_validation

# Oder nur einen Schritt
docker-compose run --rm preprocessing
```

## Troubleshooting

### Windows: Docker Socket
Wenn Docker-in-Docker nicht funktioniert, in docker-compose.yml ändern:
```yaml
# Von:
- /var/run/docker.sock:/var/run/docker.sock
# Zu:
- //var/run/docker.sock:/var/run/docker.sock
```

### Ports belegt
```bash
# Andere Ports verwenden in docker-compose.yml
ports:
  - "8081:8080"  # Airflow auf Port 8081
```

### Neustart nach Fehler
```bash
docker-compose down
docker-compose up -d
```
