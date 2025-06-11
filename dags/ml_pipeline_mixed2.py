from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.contrib.operators.docker_compose_operator import DockerComposeOperator
from airflow.utils.dates import days_ago
import os
import logging
import subprocess

# ─── Logging setup ──────────────────────────────────────────────────────────────
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
    'ml_pipeline_mixed',
    default_args=default_args,
    description='ML Pipeline: DVC + Docker services',
    schedule_interval='*/2 * * * *',      # every 2 minutes
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,                    # no overlapping runs
    tags=['ml', 'training', 'docker', 'dvc'],
) as dag:

    # ─── Constants ───────────────────────────────────────────────────────────────
    PROJECT_DIR = "/opt/airflow/project"
    COMPOSE_FILES = ["/opt/airflow/docker-compose.yml"]

    # ─── 0. Environment check ────────────────────────────────────────────────────
    def check_env_vars(**context):
        required = [
            'GITHUB_TOKEN', 'GITHUB_REPO_OWNER', 'GITHUB_REPO_NAME',
            'DAGSHUB_USER_TOKEN', 'DAGSHUB_REPO_OWNER', 'DAGSHUB_REPO_NAME'
        ]
        missing = [v for v in required if not os.getenv(v)]
        if missing:
            raise ValueError(f"Missing env vars: {missing}")
        # Docker sanity check
        result = subprocess.run(
            ['docker', 'ps'], capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            logger.error(result.stderr)
            raise RuntimeError("Docker CLI not responding")
        logger.info("All required env vars present and Docker is reachable")

    check_env = PythonOperator(
        task_id='check_environment',
        python_callable=check_env_vars,
    )

    # ─── Helper to build DockerComposeOperator tasks ─────────────────────────────
    def make_dc_task(task_id: str, service: str):
        return DockerComposeOperator(
            task_id=task_id,
            project_dir=PROJECT_DIR,
            compose_files=COMPOSE_FILES,
            services=[service],
            # tear down this service’s container(s) after run
            remove_orphans=True,
            build=False,
        )

    # ─── 1. DVC sync ─────────────────────────────────────────────────────────────
    dvc_sync = make_dc_task('dvc_sync', 'dvc-sync')

    # ─── 2. Preprocessing ────────────────────────────────────────────────────────
    preprocessing = make_dc_task('preprocessing', 'preprocessing')

    # ─── 3. Model training ──────────────────────────────────────────────────────
    model_training = make_dc_task('model_training', 'model_training')

    # ─── 4. Model validation ────────────────────────────────────────────────────
    model_validation = make_dc_task('model_validation', 'model_validation')

    # ─── 5. Tests ────────────────────────────────────────────────────────────────
    tests = make_dc_task('run_tests', 'tests')

    # ─── 6. Full cleanup ─────────────────────────────────────────────────────────
    cleanup = DockerComposeOperator(
        task_id='cleanup',
        project_dir=PROJECT_DIR,
        compose_files=COMPOSE_FILES,
        # bring down entire compose, removing orphans
        command='down --remove-orphans',
        build=False,
    )

    # ─── Dependencies ────────────────────────────────────────────────────────────
    check_env \
        >> dvc_sync \
        >> preprocessing \
        >> model_training \
        >> model_validation \
        >> tests \
        >> cleanup
