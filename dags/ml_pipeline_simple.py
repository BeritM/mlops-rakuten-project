from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Default arguments für alle Tasks
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG Definition
dag = DAG(
    'ml_pipeline_simple',
    default_args=default_args,
    description='Einfache ML Training Pipeline',
    schedule_interval=None,  # Manuell triggern
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'training', 'simple'],
)

# Python Funktion für Status-Checks
def print_status(task_name):
    """Einfache Funktion die den Status ausgibt"""
    print(f"✅ {task_name} erfolgreich abgeschlossen!")
    print(f"Timestamp: {datetime.now()}")
    return f"{task_name} completed"

# Task 1: DVC Pull
dvc_pull = BashOperator(
    task_id='dvc_pull',
    bash_command='echo "🔄 Starte DVC Pull..." && sleep 2 && echo "✅ DVC Pull abgeschlossen"',
    dag=dag,
)

# Task 2: Preprocessing
preprocessing = BashOperator(
    task_id='preprocessing',
    bash_command='echo "🔧 Starte Preprocessing..." && sleep 3 && echo "✅ Preprocessing abgeschlossen"',
    dag=dag,
)

# Task 3: Model Training
model_training = BashOperator(
    task_id='model_training',
    bash_command='echo "🤖 Starte Model Training..." && sleep 5 && echo "✅ Model Training abgeschlossen"',
    dag=dag,
)

# Task 4: Model Validation
model_validation = BashOperator(
    task_id='model_validation',
    bash_command='echo "🔍 Starte Model Validation..." && sleep 3 && echo "✅ Model Validation abgeschlossen"',
    dag=dag,
)

# Task 5: Status Report (Python Operator Beispiel)
status_report = PythonOperator(
    task_id='status_report',
    python_callable=print_status,
    op_kwargs={'task_name': 'ML Pipeline'},
    dag=dag,
)

# Task Dependencies (Reihenfolge festlegen)
dvc_pull >> preprocessing >> model_training >> model_validation >> status_report