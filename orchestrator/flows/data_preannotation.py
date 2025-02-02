from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "your_name",
    "depends_on_past": False,
    "email": ["your_email@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def hello_world():
    print("Hello World!")


with DAG(
    dag_id="image_data_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 7, 4),
    schedule_interval=None,
    catchup=False,
) as dag:
    task = PythonOperator(
        task_id="fetch_website_data",
        python_callable=hello_world,
        provide_context=True,
    )
