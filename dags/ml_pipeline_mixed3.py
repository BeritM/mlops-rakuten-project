import os
import subprocess
import logging

from datetime import timedelta
from dotenv import load_dotenv

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

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
    dag_id='ml_pipeline_mixed3',
    default_args=default_args,
    description='ML Pipeline: run DVC in PythonOperator, rest via DockerOperator',
    schedule_interval='*/2 * * * *',
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['ml','training','docker','dvc'],
) as dag:

    PROJECT_DIR = '/opt/airflow/project'
    SHARED_VOL  = '/opt/airflow/shared_volume'
    DOCKER_SOCK = 'unix://var/run/docker.sock'

    # 0. Env & Docker CLI sanity check
    def check_env(**context):
        required = [
            'GITHUB_TOKEN','GITHUB_REPO_OWNER','GITHUB_REPO_NAME',
            'DAGSHUB_USER_TOKEN','DAGSHUB_REPO_OWNER','DAGSHUB_REPO_NAME'
        ]
        missing = [v for v in required if not os.getenv(v)]
        if missing:
            raise ValueError(f"Missing env vars: {missing}")
        res = subprocess.run(['docker','ps'], capture_output=True, text=True, timeout=15)
        if res.returncode != 0:
            logger.error(res.stderr)
            raise RuntimeError("Docker CLI not responding")
        logger.info("Env OK & Docker reachable")

    check_env = PythonOperator(
        task_id='check_environment',
        python_callable=check_env,
    )

    # 1. DVC sync in‐place (no docker-compose)
    def run_dvc_pull(**context):
        import os, subprocess, logging
        log = logging.getLogger(__name__)
        os.chdir(PROJECT_DIR)
        cmd = [
        'dvc', 'pull',
        'shared_volume/data/raw',
        'shared_volume/data/processed',
        'shared_volume/data/feedback',
        '--force','--verbose',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
        log.info(result.stdout)
        if result.returncode != 0:
            log.error(result.stderr)
            raise RuntimeError(f"DVC pull failed: {result.stderr}")

    dvc_sync = PythonOperator(
        task_id='dvc_sync',
        python_callable=run_dvc_pull,
        dag=dag,
    )

    # Helper for the downstream Docker services
    def make_docker_task(task_id, image, command):
        return DockerOperator(
            task_id=task_id,
            image=image,
            api_version='auto',
            auto_remove=True,
            command=command,
            docker_url=DOCKER_SOCK,
            network_mode='bridge',
            mounts=[
                Mount(target='/app', source=PROJECT_DIR, type='bind'),
                Mount(target='/app/shared_volume', source=SHARED_VOL, type='bind'),
            ],
            working_dir='/app',
            environment={
                'GITHUB_TOKEN': os.getenv('GITHUB_TOKEN'),
                'GITHUB_REPO_OWNER': os.getenv('GITHUB_REPO_OWNER'),
                'GITHUB_REPO_NAME': os.getenv('GITHUB_REPO_NAME'),
                'DAGSHUB_USER_TOKEN': os.getenv('DAGSHUB_USER_TOKEN'),
                'DAGSHUB_REPO_OWNER': os.getenv('DAGSHUB_REPO_OWNER'),
                'DAGSHUB_REPO_NAME': os.getenv('DAGSHUB_REPO_NAME'),
                'PYTHONPATH': '/app',
                'PYTHONUNBUFFERED': '1',
            },
            force_pull=False,
        )

    preprocessing = make_docker_task(
        'preprocessing',
        'mlops-rakuten-project-preprocessing:latest',
        'python plugins/cd4ml/data_processing/run_preprocessing.py'
    )

    model_training = make_docker_task(
        'model_training',
        'mlops-rakuten-project-model_training:latest',
        'python plugins/cd4ml/model_training/run_model_training.py'
    )

    model_validation = make_docker_task(
        'model_validation',
        'mlops-rakuten-project-model_validation:latest',
        'python plugins/cd4ml/model_validation/run_model_validation.py'
    )

    run_tests = make_docker_task(
        'run_tests',
        'mlops-rakuten-project-tests:latest',
        'pytest plugins/cd4ml/tests/test_predict_service.py -v -rA'
    )

    cleanup = PythonOperator(
        task_id='cleanup',
        python_callable=lambda: subprocess.run(
            ['docker','compose','-f',f'{PROJECT_DIR}/docker-compose.yml','down','--remove-orphans'],
            cwd=PROJECT_DIR
        ),
        trigger_rule='all_done',
    )

    # DAG ordering
    check_env >> dvc_sync >> preprocessing >> model_training >> model_validation >> run_tests >> cleanup
