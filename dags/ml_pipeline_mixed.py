from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import os
import logging
import subprocess

# Logging setup
logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# DAG Definition
dag = DAG(
    'ml_pipeline_mixed',
    default_args=default_args,
    description='ML Pipeline mit gemischtem Ansatz - DVC direkt, Rest über Docker',
    schedule_interval=None,  # Manuell triggern
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'training', 'docker', 'dvc'],
)

# Projektverzeichnis im Airflow Container
WORKDIR = "/opt/airflow/project"

def check_environment_variables(**context):
    """Überprüfe ob notwendige Environment Variables gesetzt sind"""
    required_vars = ['GITHUB_TOKEN', 'GITHUB_REPO_OWNER', 'GITHUB_REPO_NAME']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    logger.info("All required environment variables are set")
    logger.info(f"GITHUB_REPO_OWNER: {os.getenv('GITHUB_REPO_OWNER')}")
    logger.info(f"GITHUB_REPO_NAME: {os.getenv('GITHUB_REPO_NAME')}")
    
    # Check docker access
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Docker command failed: {result.stderr}")
            raise Exception("Cannot access Docker. Check permissions.")
        logger.info("Docker access confirmed")
    except Exception as e:
        logger.error(f"Docker check failed: {e}")
        raise

def run_dvc_sync(**context):
    """Run DVC sync directly in Airflow container"""
    import subprocess
    import os
    
    os.chdir(WORKDIR)
    
    # Set git safe directory
    subprocess.run(['git', 'config', '--global', '--add', 'safe.directory', WORKDIR])
    
    # Set environment variables
    env = os.environ.copy()
    env['DAGSHUB_USER_TOKEN'] = os.getenv('DAGSHUB_USER_TOKEN', '')
    env['DAGSHUB_REPO_OWNER'] = os.getenv('DAGSHUB_REPO_OWNER', '')
    env['DAGSHUB_REPO_NAME'] = os.getenv('DAGSHUB_REPO_NAME', '')
    
    # Run DVC pull
    result = subprocess.run(
        ['dvc', 'pull', 'shared_volume/data/raw', 'shared_volume/data/processed', 
         'shared_volume/data/feedback', 'shared_volume/models', '--force'],
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"DVC pull failed: {result.stderr}")
        raise Exception(f"DVC pull failed: {result.stderr}")
    
    logger.info(f"DVC pull successful: {result.stdout}")

# Task 0 - Environment Check
check_env = PythonOperator(
    task_id="check_environment_variables",
    python_callable=check_environment_variables,
    dag=dag,
)

# Task 1 - DVC Sync (Python Operator)
dvc_sync = PythonOperator(
    task_id="dvc_sync",
    python_callable=run_dvc_sync,
    dag=dag,
)

# Task 2 - Preprocessing (mit --no-deps um Dependencies zu ignorieren)
preprocessing = BashOperator(
    task_id="preprocessing",
    bash_command=f"""
    cd {WORKDIR} && \
    echo "Starting preprocessing..." && \
    docker-compose run --rm --no-deps \
        -e GITHUB_TOKEN=${{GITHUB_TOKEN}} \
        -e GITHUB_REPO_OWNER=${{GITHUB_REPO_OWNER}} \
        -e GITHUB_REPO_NAME=${{GITHUB_REPO_NAME}} \
        preprocessing && \
    echo "Preprocessing completed successfully"
    """,
    dag=dag,
)

# Task 3 - Model Training
model_training = BashOperator(
    task_id="model_training",
    bash_command=f"""
    cd {WORKDIR} && \
    echo "Starting model training..." && \
    docker-compose run --rm --no-deps \
        -e GITHUB_TOKEN=${{GITHUB_TOKEN}} \
        -e GITHUB_REPO_OWNER=${{GITHUB_REPO_OWNER}} \
        -e GITHUB_REPO_NAME=${{GITHUB_REPO_NAME}} \
        model_training && \
    echo "Model training completed successfully"
    """,
    dag=dag,
)

# Task 4 - Model Validation
model_validation = BashOperator(
    task_id="model_validation",
    bash_command=f"""
    cd {WORKDIR} && \
    echo "Starting model validation..." && \
    docker-compose run --rm --no-deps \
        -e GITHUB_TOKEN=${{GITHUB_TOKEN}} \
        -e GITHUB_REPO_OWNER=${{GITHUB_REPO_OWNER}} \
        -e GITHUB_REPO_NAME=${{GITHUB_REPO_NAME}} \
        model_validation && \
    echo "Model validation completed successfully"
    """,
    dag=dag,
)

# Task 5 - Tests (optional)
run_tests = BashOperator(
    task_id="run_tests",
    bash_command=f"""
    cd {WORKDIR} && \
    echo "Skipping tests for now..." && \
    echo "Tests would run here"
    """,
    dag=dag,
    trigger_rule='none_failed_min_one_success',
)

# Task 6 - Cleanup
cleanup = BashOperator(
    task_id="cleanup",
    bash_command=f"""
    cd {WORKDIR} && \
    echo "Cleaning up Docker resources..." && \
    docker-compose down || true && \
    docker container prune -f && \
    docker image prune -f && \
    echo "Cleanup completed"
    """,
    dag=dag,
    trigger_rule='all_done',
)

# Pipeline Definition
check_env >> dvc_sync >> preprocessing >> model_training >> model_validation >> run_tests >> cleanup