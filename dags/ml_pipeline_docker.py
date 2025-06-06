from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# Default arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG Definition
dag = DAG(
    'ml_pipeline_docker',
    default_args=default_args,
    description='ML Pipeline mit Docker Compose',
    schedule_interval=None,  # Manuell triggern
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'training', 'docker'],
)

# Docker Compose 

WORKDIR = "/opt/airflow/project"

# Task 1 ─ DVC Sync
dvc_sync = BashOperator(
    task_id="dvc_sync",
    bash_command=f"cd {WORKDIR} && docker compose run --rm dvc-sync",
    dag=dag,
)

# Task 2 ─ Preprocessing
preprocessing = BashOperator(
    task_id="preprocessing",
    bash_command=f"cd {WORKDIR} && docker compose run --rm preprocessing",
    dag=dag,
)

# Task 3 ─ Model Training
model_training = BashOperator(
    task_id="model_training",
    bash_command=f"cd {WORKDIR} && docker compose run --rm model_training",
    dag=dag,
)

# Task 4 ─ Model Validation
model_validation = BashOperator(
    task_id="model_validation",
    bash_command=f"cd {WORKDIR} && docker compose run --rm model_validation",
    dag=dag,
)

# Task 5 ─ Tests
run_tests = BashOperator(
    task_id="run_tests",
    bash_command=f"cd {WORKDIR} && docker compose run --rm tests",
    dag=dag,
)

# Pipeline 
dvc_sync >> preprocessing >> model_training >> model_validation >> run_tests