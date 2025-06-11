import os
import subprocess
import logging

from datetime import timedelta
from dotenv import load_dotenv

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# ─── Load .env ────────────────────────────────────────────────────────────────
load_dotenv('/opt/airflow/.env')

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='ml_pipeline_mixed_final',
    default_args=default_args,
    description='ML Pipeline entirely driven via docker-compose',
    schedule_interval='*/2 * * * *',    # every 2 minutes
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['ml','training','docker','dvc'],
) as dag:

    PROJECT_DIR = '/opt/airflow/project'
    COMPOSE_YML = f'{PROJECT_DIR}/docker-compose.yml'
    COMPOSE = f'cd {PROJECT_DIR} && docker-compose -f {COMPOSE_YML}'

    # 0. Env & Docker CLI check
    def check_env(**context):
        required = [
            'GITHUB_TOKEN','GITHUB_REPO_OWNER','GITHUB_REPO_NAME',
            'DAGSHUB_USER_TOKEN','DAGSHUB_REPO_OWNER','DAGSHUB_REPO_NAME'
        ]
        missing = [v for v in required if not os.getenv(v)]
        if missing:
            raise ValueError(f"Missing env vars: {missing}")

        res = subprocess.run(
            ['docker','ps'], capture_output=True, text=True, timeout=15
        )
        if res.returncode != 0:
            logger.error(res.stderr)
            raise RuntimeError("Docker CLI not responding")
        logger.info("Env OK & Docker reachable")

    check_env_task = PythonOperator(
        task_id='check_environment',
        python_callable=check_env,
    )

    # 1. DVC sync
    dvc_sync = BashOperator(
        task_id='dvc_sync',
        bash_command=(
            COMPOSE
            + ' run --rm --no-deps '
              '-e DAGSHUB_USER_TOKEN=$DAGSHUB_USER_TOKEN '
              '-e DAGSHUB_REPO_OWNER=$DAGSHUB_REPO_OWNER '
              '-e DAGSHUB_REPO_NAME=$DAGSHUB_REPO_NAME '
              'dvc-sync'
        ),
    )

    # 2. Preprocessing
    preprocessing = BashOperator(
        task_id='preprocessing',
        bash_command=f'{COMPOSE} run --rm --no-deps preprocessing',
    )

    # 3. Model training
    model_training = BashOperator(
        task_id='model_training',
        bash_command=f'{COMPOSE} run --rm --no-deps model_training',
    )

    # 4. Model validation
    model_validation = BashOperator(
        task_id='model_validation',
        bash_command=f'{COMPOSE} run --rm --no-deps model_validation',
    )

    # 5. Tests
    run_tests = BashOperator(
        task_id='run_tests',
        bash_command=f'{COMPOSE} run --rm --no-deps tests',
        trigger_rule='none_failed_min_one_success',
    )

    # 6. Cleanup
    cleanup = BashOperator(
        task_id='cleanup',
        bash_command=f'{COMPOSE} down --remove-orphans',
        trigger_rule='all_done',
    )

    check_env_task \
        >> dvc_sync \
        >> preprocessing \
        >> model_training \
        >> model_validation \
        >> run_tests \
        >> cleanup
