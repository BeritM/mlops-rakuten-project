import os
import subprocess
import logging

from datetime import timedelta
from dotenv import load_dotenv

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# ─── Load .env ────────────────────────────────────────────────────────────────
load_dotenv('/opt/airflow/.env')

# ─── Validate Host Paths ───────────────────────────────────────────────────────
HOST_PROJECT_PATH = os.getenv('HOST_PROJECT_PATH')
if not HOST_PROJECT_PATH:
    raise RuntimeError('Please define HOST_PROJECT_PATH in the environment!')
HOST_SHARED_VOL = os.path.join(HOST_PROJECT_PATH, 'shared_volume')

logger = logging.getLogger(__name__)

def build_default_args():
    return {
        'owner': 'mlops-team',
        'depends_on_past': False,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
    }

with DAG(
    dag_id='ml_pipeline_4',
    default_args=build_default_args(),
    description='ML Pipeline: run via BashOperator for DVC sync, rest via DockerOperator',
    schedule_interval='*/2 * * * *',
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=2,  # allow overlapping runs
    tags=['ml', 'training', 'docker', 'dvc'],
) as dag:

    PROJECT_DIR = '/opt/airflow/project'
    SHARED_VOL = '/opt/airflow/shared_volume'

    # 0. Env & Docker CLI sanity check
    def check_env(**context):
        required = [
            'GITHUB_TOKEN', 'GITHUB_REPO_OWNER', 'GITHUB_REPO_NAME',
            'DAGSHUB_USER_TOKEN', 'DAGSHUB_REPO_OWNER', 'DAGSHUB_REPO_NAME'
        ]
        missing = [v for v in required if not os.getenv(v)]
        if missing:
            raise ValueError(f'Missing env vars: {missing}')
        res = subprocess.run(['docker', 'ps'], capture_output=True, text=True, timeout=15)
        if res.returncode != 0:
            logger.error(res.stderr)
            raise RuntimeError('Docker CLI not responding')
        logger.info('Env OK & Docker reachable')

    check_env = PythonOperator(
        task_id='check_environment',
        python_callable=check_env,
    )

    # 1. DVC sync via BashOperator
    dvc_sync = BashOperator(
        task_id='dvc_sync',
        bash_command=(
            f'cd {PROJECT_DIR} && '
            'dvc pull '
            'shared_volume/data/raw '
            'shared_volume/data/processed '
            'shared_volume/data/feedback '
            'shared_volume/models '
            '--force --verbose || true'
        ),
        execution_timeout=timedelta(minutes=10),
        retries=0,
    )

        # 2. Preprocessing via BashOperator
    preprocessing = BashOperator(
        task_id='preprocessing',
        bash_command=(
            f'cd {PROJECT_DIR} && '
            'python plugins/cd4ml/data_processing/run_preprocessing.py || true'
        ),
        execution_timeout=timedelta(minutes=20),
        retries=0,
    )

    # 3. Model Training via BashOperator
    model_training = BashOperator(
        task_id='model_training',
        bash_command=(
            f'cd {PROJECT_DIR} && '
            'python plugins/cd4ml/model_training/run_model_training.py || true'
        ),
        execution_timeout=timedelta(minutes=30),
        retries=0,
    )

    # 4. Model Validation via BashOperator
    model_validation = BashOperator(
        task_id='model_validation',
        bash_command=(
            f'cd {PROJECT_DIR} && '
            'python plugins/cd4ml/model_validation/run_model_validation.py || true'
        ),
        execution_timeout=timedelta(minutes=10),
        retries=0,
    )

    # 5. Run Tests via BashOperator 
    run_tests = BashOperator(
        task_id='run_tests',
        bash_command=(
            f'cd {PROJECT_DIR} && '
            'pytest plugins/cd4ml/tests/test_predict_service.py -v -rA || true'
        ),
        execution_timeout=timedelta(minutes=10),
        retries=0,
    )

    # 6. Cleanup via BashOperator
    cleanup = BashOperator(
        task_id='cleanup',
        bash_command=(
            f'cd {PROJECT_DIR} && '
            'docker-compose -f docker-compose.yml down --remove-orphans || true'
        ),
        trigger_rule='all_done',
        retries=0,
    )

    # DAG ordering
    check_env >> dvc_sync >> preprocessing >> model_training >> model_validation >> run_tests >> cleanup
