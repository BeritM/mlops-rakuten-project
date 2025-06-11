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

# ─── Logger ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Default args ──────────────────────────────────────────────────────────────
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# ─── DAG Definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id='ml_pipeline_mixed4',
    default_args=default_args,
    description='ML Pipeline: DVC + DockerOperator (cache bind-mount, verbose, host net)',
    schedule_interval='*/2 * * * *',
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'training', 'docker', 'dvc'],
) as dag:

    PROJECT_DIR = '/opt/airflow/project'
    SHARED_VOL  = '/opt/airflow/shared_volume'
    DOCKER_SOCK = 'unix://var/run/docker.sock'

    # ─── 0. Env & Docker sanity ─────────────────────────────────────────────────
    def check_env(**context):
        required = [
            'GITHUB_TOKEN','GITHUB_REPO_OWNER','GITHUB_REPO_NAME',
            'DAGSHUB_USER_TOKEN','DAGSHUB_REPO_OWNER','DAGSHUB_REPO_NAME'
        ]
        missing = [v for v in required if not os.getenv(v)]
        if missing:
            raise ValueError(f"Missing env vars: {missing}")

        res = subprocess.run(
            ['docker', 'ps'], capture_output=True, text=True, timeout=15
        )
        if res.returncode != 0:
            logger.error(res.stderr)
            raise RuntimeError("Docker CLI not responding")
        logger.info("Env OK & Docker reachable")

    check_env_task = PythonOperator(
        task_id='check_environment',
        python_callable=check_env,
    )

    # ─── Docker task helper ─────────────────────────────────────────────────────
    def make_docker_task(task_id, image, cmd, mounts=None, extra_env=None, network='bridge'):
        base_mounts = [
            Mount(target='/app', source=PROJECT_DIR, type='bind'),
            Mount(target='/app/shared_volume', source=SHARED_VOL, type='bind'),
        ]
        mounts = (base_mounts + mounts) if mounts else base_mounts

        env = {
            'GITHUB_TOKEN': os.getenv('GITHUB_TOKEN'),
            'GITHUB_REPO_OWNER': os.getenv('GITHUB_REPO_OWNER'),
            'GITHUB_REPO_NAME': os.getenv('GITHUB_REPO_NAME'),
            'DAGSHUB_USER_TOKEN': os.getenv('DAGSHUB_USER_TOKEN'),
            'DAGSHUB_REPO_OWNER': os.getenv('DAGSHUB_REPO_OWNER'),
            'DAGSHUB_REPO_NAME': os.getenv('DAGSHUB_REPO_NAME'),
            'PYTHONPATH': '/app',
            'PYTHONUNBUFFERED': '1',
        }
        if extra_env:
            env.update(extra_env)

        return DockerOperator(
            task_id=task_id,
            image=image,
            api_version='auto',
            auto_remove=True,
            command=cmd,
            docker_url=DOCKER_SOCK,
            network_mode=network,
            mounts=mounts,
            working_dir='/app',
            environment=env,
        )

    # ─── 1. DVC Sync ──────────────────────────────────────────────────────────────
    dvc_sync = make_docker_task(
        task_id='dvc_sync',
        image='mlops-rakuten-project-dvc-sync:latest',
        cmd=(
            'dvc pull '
            'shared_volume/data/raw shared_volume/data/processed '
            'shared_volume/data/feedback shared_volume/models '
            '--force --verbose'
        ),
        mounts=[
            # bind-mount the host project’s cache directory
            Mount(target='/app/.dvc/cache', source=f'{PROJECT_DIR}/.dvc/cache', type='bind'),
        ],
        network='host',  # allows container to use host DNS/network
    )

    # ─── 2. Preprocessing ────────────────────────────────────────────────────────
    preprocessing = make_docker_task(
        task_id='preprocessing',
        image='preprocessing:latest',
        cmd='python plugins/cd4ml/data_processing/run_preprocessing.py'
    )

    # ─── 3. Model Training ───────────────────────────────────────────────────────
    model_training = make_docker_task(
        task_id='model_training',
        image='model_training:latest',
        cmd='python plugins/cd4ml/model_training/run_model_training.py'
    )

    # ─── 4. Model Validation ─────────────────────────────────────────────────────
    model_validation = make_docker_task(
        task_id='model_validation',
        image='model_validation:latest',
        cmd='python plugins/cd4ml/model_validation/run_model_validation.py'
    )

    # ─── 5. Tests ────────────────────────────────────────────────────────────────
    run_tests = make_docker_task(
        task_id='run_tests',
        image='tests:latest',
        cmd='pytest plugins/cd4ml/tests/test_predict_service.py -v -rA'
    )

    # ─── 6. Cleanup ──────────────────────────────────────────────────────────────
    cleanup = DockerOperator(
        task_id='cleanup',
        image='docker:latest',
        api_version='auto',
        auto_remove=True,
        command='sh -c "docker-compose -f /app/docker-compose.yml down --remove-orphans"',
        docker_url=DOCKER_SOCK,
        network_mode='bridge',
        mounts=[
            Mount(target='/app', source=PROJECT_DIR, type='bind'),
            Mount(target='/var/run/docker.sock', source='/var/run/docker.sock', type='bind'),
        ],
    )

    # ─── Flow ────────────────────────────────────────────────────────────────────
    check_env_task \
        >> dvc_sync \
        >> preprocessing \
        >> model_training \
        >> model_validation \
        >> run_tests \
        >> cleanup
