import os
import subprocess
import logging
import stat
import pwd
import grp

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
    dag_id='ml_pipeline_mixed_current',
    default_args=default_args,
    description='ML Pipeline: run DVC in PythonOperator, rest via DockerOperator',
    schedule_interval='*/10 * * * *',
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['ml','training','docker','dvc'],
) as dag:

    PROJECT_DIR = os.environ["HOST_PROJECT_PATH"] # ← host path	
    SHARED_VOL = os.path.join(PROJECT_DIR, "shared_volume")

    PROJECT_DIR_DVC = '/opt/airflow/project'
    DOCKER_SOCK = 'unix:///var/run/docker.sock'

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

    # 1. DVC sync with proper permission handling
    def run_dvc_pull(**context):
        import os
        import subprocess
        import logging
        import stat
        import pwd
        import grp
        from pathlib import Path
        
        log = logging.getLogger(__name__)
        
        # Change to project directory
        os.chdir(PROJECT_DIR_DVC)
        log.info(f"Changed to directory: {os.getcwd()}")
        
        # Ensure .dvc directory structure exists
        dvc_dir = Path(PROJECT_DIR_DVC) / '.dvc'
        cache_dir = dvc_dir / 'cache'
        files_dir = cache_dir / 'files'
        md5_dir = files_dir / 'md5'
        
        # Create directories with proper permissions
        for directory in [dvc_dir, cache_dir, files_dir, md5_dir]:
            directory.mkdir(parents=True, exist_ok=True, mode=0o755)
            log.info(f"Ensured directory exists: {directory}")
        
        # Try to get airflow user info
        try:
            airflow_user = pwd.getpwnam('airflow')
            airflow_uid = airflow_user.pw_uid
            airflow_gid = airflow_user.pw_gid
            log.info(f"Airflow user: uid={airflow_uid}, gid={airflow_gid}")
        except KeyError:
            # Fallback to current user
            airflow_uid = os.getuid()
            airflow_gid = os.getgid()
            log.info(f"Using current user: uid={airflow_uid}, gid={airflow_gid}")
        
        # Set ownership and permissions recursively
        def fix_permissions(path):
            try:
                os.chown(path, airflow_uid, airflow_gid)
                if os.path.isdir(path):
                    os.chmod(path, 0o755)
                    for item in os.listdir(path):
                        fix_permissions(os.path.join(path, item))
                else:
                    os.chmod(path, 0o644)
            except PermissionError as e:
                log.warning(f"Could not fix permissions for {path}: {e}")
            except Exception as e:
                log.warning(f"Error fixing permissions for {path}: {e}")
        
        # Fix permissions for DVC directories
        fix_permissions(str(dvc_dir))
        log.info("Fixed permissions for .dvc directory")
        
        # Clear any existing problematic cache files
        try:
            import shutil
            temp_files = list(cache_dir.rglob("*.tmp"))
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    log.info(f"Removed temp file: {temp_file}")
                except Exception as e:
                    log.warning(f"Could not remove temp file {temp_file}: {e}")
        except Exception as e:
            log.warning(f"Error cleaning temp files: {e}")
        
        # Prepare environment for DVC
        env = os.environ.copy()
        env['DVC_CACHE_DIR'] = str(cache_dir)
        
        # DVC pull command with better error handling
        cmd = [
            'dvc', 'pull', 
            '--force',
            '--verbose',
        ]
        
        log.info(f"Running DVC command: {' '.join(cmd)}")
        log.info(f"Working directory: {os.getcwd()}")
        log.info(f"Cache directory: {cache_dir}")
        
        # Run DVC pull
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                env=env,
                timeout=600  # 10 minute timeout
            )
            
            # Log output
            if result.stdout:
                log.info("DVC STDOUT:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        log.info(line)
            
            if result.stderr:
                log.info("DVC STDERR:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        log.info(line)
            
            if result.returncode != 0:
                log.error(f"DVC pull failed with return code: {result.returncode}")
                raise RuntimeError(f"DVC pull failed: {result.stderr}")
            
            log.info("DVC pull completed successfully")
            
        except subprocess.TimeoutExpired:
            log.error("DVC pull timed out after 10 minutes")
            raise RuntimeError("DVC pull timed out")
        except Exception as e:
            log.error(f"DVC pull failed with exception: {e}")
            raise

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
                'DAGSHUB_USER_NAME': os.getenv('DAGSHUB_USER_TOKEN'),
                'DAGSHUB_USER_TOKEN': os.getenv('DAGSHUB_USER_TOKEN'),
                'DAGSHUB_REPO_OWNER': os.getenv('DAGSHUB_REPO_OWNER'),
                'DAGSHUB_REPO_NAME': os.getenv('DAGSHUB_REPO_NAME'),
                'DATA_RAW_DIR': os.getenv('DATA_RAW_DIR'),
                'DATA_PROCESSED_DIR': os.getenv('DATA_PROCESSED_DIR'),
                'MODEL_DIR': os.getenv('MODEL_DIR'),
                'DATA_FEEDBACK_DIR': os.getenv('DATA_FEEDBACK_DIR'),
                'X_RAW': os.getenv('X_RAW'),
                'Y_RAW': os.getenv('Y_RAW'),
                'X_Y_RAW': os.getenv('X_Y_RAW'),
                'X_TRAIN_TFIDF': os.getenv('X_TRAIN_TFIDF'),
                'X_VALIDATE_TFIDF': os.getenv('X_VALIDATE_TFIDF'),
                'X_TEST_TFIDF': os.getenv('X_TEST_TFIDF'),
                'Y_TRAIN': os.getenv('Y_TRAIN'),
                'Y_VALIDATE': os.getenv('Y_VALIDATE'),
                'Y_TEST': os.getenv('Y_TEST'),
                'TFIDF_VECTORIZER': os.getenv('TFIDF_VECTORIZER'),
                'MODEL': os.getenv('MODEL'),
                'PRODUCT_DICTIONARY': os.getenv('PRODUCT_DICTIONARY'),
                'CLASS_REPORT': os.getenv('CLASS_REPORT'),
                'CLASS_REPORT_VALIDATION': os.getenv('CLASS_REPORT_VALIDATION'),
                'CURRENT_RUN_ID': os.getenv('CURRENT_RUN_ID'),
                'PARAM_CONFIG': os.getenv('PARAM_CONFIG'),
                'FEEDBACK_CSV': os.getenv('FEEDBACK_CSV'),
                'RETRAIN_RAW': os.getenv('RETRAIN_RAW'),
                'PYTHONPATH': '/app',
                'PYTHONUNBUFFERED': '1', 
            },
            force_pull=False,
            mount_tmp_dir=False,
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

    # DAG ordering
    check_env >> dvc_sync >> preprocessing >> model_training >> model_validation