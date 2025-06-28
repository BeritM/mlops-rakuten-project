from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

with DAG(
    dag_id='dvc_cache_cleaner',
    start_date=datetime(2025, 1, 1),
    schedule_interval=timedelta(weeks=1), # once every week
    catchup=False,
    tags=['dvc', 'maintenance'],
) as dag:
    clean_dvc_cache = BashOperator(
        task_id='clean_dvc_cache',
        bash_command="dvc gc -w --rev HEAD",
    )
