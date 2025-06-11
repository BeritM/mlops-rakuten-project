import os
import subprocess
import logging

from datetime import timedelta
from dotenv import load_dotenv

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# ─── Load .env ────────────────────────────────────────────────────────────────
load_dotenv('/opt/airflow/.env')
logger = logging.getLogger(__name__)

# This must point to your Mac’s absolute project path, and be shared in Docker Desktop
HOST_PROJECT = os.getenv('HOST_PROJECT_PATH')
if not HOST_PROJECT:
    raise RuntimeError("HOST_PROJECT_PATH must be set to your host repo path")

COMPOSE_CMD = (
    f'cd {HOST_PROJECT} && '
    f'docker-compose -f {HOST_PROJECT}/docker-compose.yml'
)

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='ml_pipeline_mixed2',
    default_args=default_args,
    description='ML Pipeline via BashOperator + host‐path docker-compose',
    schedule_interval='*/2 * * * *',
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['ml','training','docker','dvc'],
) as dag:

    # 0. Env & Docker check
    def check_env(**context):
        required = [
            'GITHUB_TOKEN','GITHUB_REPO_OWNER','GITHUB_REPO_NAME',
            'DAGSHUB_USER_TOKEN','DAGSHUB_REPO_OWNER','DAGSHUB_REPO_NAME'
        ]
        missing = [v for v in required if not os.getenv(v)]
        if missing:
            raise ValueError(f"Missing env vars: {missing}")
        r = subprocess.run(['docker','ps'], capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            logger.error(r.stderr)
            raise RuntimeError("Docker CLI not responding")
        logger.info("Docker CLI OK")

    check_env = PythonOperator(
        task_id='check_environment',
        python_callable=check_env,
    )

    # 1. DVC sync (in-place, runs on the host‐mounted folder)
    def run_dvc_pull(**context):
        os.chdir(HOST_PROJECT)
        r = subprocess.run(
            ['dvc','pull','--force','--verbose'],
            capture_output=True, text=True, env=os.environ.copy()
        )
        logger.info(r.stdout)
        if r.returncode != 0:
            logger.error(r.stderr)
            raise RuntimeError("DVC pull failed")

    dvc_sync = PythonOperator(
        task_id='dvc_sync',
        python_callable=run_dvc_pull,
    )

    # 2. Preprocessing via host docker-compose
    preprocessing = BashOperator(
        task_id='preprocessing',
        bash_command=(
            COMPOSE_CMD +
            " run --rm --no-deps "
            "-e GITHUB_TOKEN=$GITHUB_TOKEN "
            "-e GITHUB_REPO_OWNER=$GITHUB_REPO_OWNER "
            "-e GITHUB_REPO_NAME=$GITHUB_REPO_NAME "
            "preprocessing"
        ),
    )

    # 3. Model Training
    model_training = BashOperator(
        task_id='model_training',
        bash_command=(
            COMPOSE_CMD +
            " run --rm --no-deps "
            "-e GITHUB_TOKEN=$GITHUB_TOKEN "
            "-e GITHUB_REPO_OWNER=$GITHUB_REPO_OWNER "
            "-e GITHUB_REPO_NAME=$GITHUB_REPO_NAME "
            "model_training"
        ),
    )

    # 4. Model Validation
    model_validation = BashOperator(
        task_id='model_validation',
        bash_command=(
            COMPOSE_CMD +
            " run --rm --no-deps "
            "-e GITHUB_TOKEN=$GITHUB_TOKEN "
            "-e GITHUB_REPO_OWNER=$GITHUB_REPO_OWNER "
            "-e GITHUB_REPO_NAME=$GITHUB_REPO_NAME "
            "model_validation"
        ),
    )

    # 5. Tests
    run_tests = BashOperator(
        task_id='run_tests',
        bash_command=COMPOSE_CMD + " run --rm --no-deps tests",
        trigger_rule='none_failed_min_one_success',
    )

    # 6. Cleanup
    cleanup = BashOperator(
        task_id='cleanup',
        bash_command=COMPOSE_CMD + " down --remove-orphans",
        trigger_rule='all_done',
    )

    # Dependencies
    check_env >> dvc_sync >> preprocessing >> model_training >> model_validation >> run_tests >> cleanup
