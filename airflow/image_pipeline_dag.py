from airflow import DAG
from airflow.operators.bash import BashOperator

from datetime import datetime

default_args = {
    "owner": "sahil",
    "start_date": datetime(2025, 1, 1)
}

dag = DAG(
    "image_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False
)

validate_task = BashOperator(
    task_id="validate_images",
    bash_command="python pipeline/validation.py",
    dag=dag
)

detect_task = BashOperator(
    task_id="detect_objects",
    bash_command="python pipeline/detection.py",
    dag=dag
)

validate_task >> detect_task