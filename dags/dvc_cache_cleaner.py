from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

with DAG(
    dag_id='dvc_cache_cleaner',
    start_date=datetime(2025, 1, 1),
    schedule_interval=timedelta(weeks=1), # Z.B. einmal pro Woche
    catchup=False,
    tags=['dvc', 'maintenance'],
) as dag:
    clean_dvc_cache = BashOperator(
        task_id='clean_dvc_cache',
        bash_command="dvc gc -w --rev HEAD",
        # optional parameters:
        # -w or --workspace: Only deletes unreferenced data in the current workspace
        # --rev HEAD: Keeps data referenced by HEAD commits
        # --all-tags, --all-branches: Keeps data referenced by all tags/branches
        # --all-experiments: if you are using DVC experiments
    )
