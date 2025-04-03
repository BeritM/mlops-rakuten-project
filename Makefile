# The purpose of the Makerfile is to automate the setup and configuration 
# of a machine learning project using DVC (Data Version Control) and Docker Compose.
# It includes tasks for installing dependencies, initializing Git and DVC, 
# building Docker images, running pipelines, and tracking changes.

.PHONY: uv set-dvc init-git init-dvc build run track dvc-docto all

# Installiert das uv-Tool (von Astral)
uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

# Konfiguriert den DVC Remote mithilfe der Umgebungsvariablen
set-dvc:
	dvc remote add origin s3://dvc
	dvc remote modify origin endpointurl https://dagshub.com/BeritM/mlops-rakuten-project.s3
	dvc remote modify origin --local access_key_id $(DVC_USER)
	dvc remote modify origin --local secret_access_key $(DVC_PASSWORD)
	dvc remote default origin

# Initialisiert ein neues Git-Repository und committe die initiale Struktur
init-git:
	git init
	git add .gitignore Dockerfile docker-compose-dvc.yml dvc.yaml src/
	git commit -m "Initial project structure and pipeline definition"

# Initialisiert DVC und committe die DVC-Konfigurationsdateien
init-dvc:
	dvc init
	git add .dvc/config .dvc/.gitignore
	git commit -m "Initialize DVC"

# Baut die Docker Compose Services
build:
	docker compose build

# Führt die Pipeline im dedizierten dvc-runner Service aus
run:
	docker compose run --rm dvc-runner dvc repro

# Committet Updates in Git, committe DVC-Änderungen und pusht sie
track:
	git add .
	git commit -m "updating the run"
	dvc commit
	git push
	dvc push

# Führt einen DVC-Diagnose-Check über den dvc-runner Service aus
dvc-docto:
	docker compose run --rm dvc-runner dvc doctor

# Führt alle wesentlichen Schritte nacheinander aus
all: init-git init-dvc build run
