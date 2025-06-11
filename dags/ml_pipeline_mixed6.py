import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta
from dotenv import dotenv_values

# Path to your .env file (relative to your airflow/dags/ directory or use absolute path)
ENV_PATH = "/opt/airflow/.env"

# Load variables
env_vars = dotenv_values(ENV_PATH)
# Also merge in actual env vars from Airflow's container for secrets not in .env
env_vars.update({k: v for k, v in os.environ.items() if k not in env_vars})

PROJECT_ROOT = "/opt/airflow/project"   # Your repo root, as mounted in docker-compose
SHARED_VOLUME = "/opt/airflow/shared_volume"

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    "ml_pipeline_dvc_preprocess_train_validate",
    default_args=default_args,
    description="Run DVC pull, preprocessing, training, validation every 2 minutes",
    schedule_interval="*/2 * * * *",  # every 2 minutes
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
) as dag:

    dvc_sync = DockerOperator(
        task_id="dvc_sync",
        image="mlops-rakuten-project-dvc-sync:latest",  # Adjust to your image/tag name
        api_version="auto",
        auto_remove=True,
        command=["dvc", "pull", "shared_volume/data/raw", "shared_volume/data/processed", "shared_volume/data/feedback", "shared_volume/models", "--force"],
        working_dir="/app",
        volumes=[
            f"{PROJECT_ROOT}:/app",
            f"{PROJECT_ROOT}/.git:/app/.git:ro",
            f"{SHARED_VOLUME}:/app/shared_volume",
        ],
        environment=env_vars,
        network_mode="bridge",
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
    )

    preprocessing = DockerOperator(
        task_id="preprocessing",
        image="yourproject_preprocessing:latest",
        api_version="auto",
        auto_remove=True,
        command="python plugins/cd4ml/data_processing/run_preprocessing.py",
        working_dir="/app",
        volumes=[
            f"{PROJECT_ROOT}:/app",
            f"{SHARED_VOLUME}:/app/shared_volume",
        ],
        environment=env_vars,
        network_mode="bridge",
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
    )

    model_training = DockerOperator(
        task_id="model_training",
        image="yourproject_model_training:latest",
        api_version="auto",
        auto_remove=True,
        command="python plugins/cd4ml/model_training/run_model_training.py",
        working_dir="/app",
        volumes=[
            f"{PROJECT_ROOT}:/app",
            f"{SHARED_VOLUME}:/app/shared_volume",
        ],
        environment=env_vars,
        network_mode="bridge",
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
    )

    model_validation = DockerOperator(
        task_id="model_validation",
        image="yourproject_model_validation:latest",
        api_version="auto",
        auto_remove=True,
        command="python plugins/cd4ml/model_validation/run_model_validation.py",
        working_dir="/app",
        volumes=[
            f"{PROJECT_ROOT}:/app",
            f"{SHARED_VOLUME}:/app/shared_volume",
        ],
        environment=env_vars,
        network_mode="bridge",
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
    )

    dvc_sync >> preprocessing >> model_training >> model_validation
