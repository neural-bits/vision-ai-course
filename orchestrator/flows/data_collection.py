import asyncio
import os
from datetime import datetime, timedelta

# from loguru import logger
from plugins.common.models import CommonMediaDocument
from plugins.etl.collection.producers import PexelsProcessor, UnsplashProcessor

from airflow import DAG
from airflow.operators.python import PythonOperator

# logger = logger.bind(name="RawDataCollection")

IMAGES_SUBJECT: str = os.getenv("API_IMAGE_SUBJECT")
NUM_PAGES: int = int(os.getenv("API_IMAGE_NUM_PAGES", 1))
START_PAGE: int = int(os.getenv("API_IMAGE_LAST_PAGE", 1))
NUM_ITEMS_PER_PAGE: int = int(os.getenv("API_IMAGE_NUM_ITEMS_PER_PAGE", 1))
NUM_CONSUMERS: int = int(os.getenv("NUM_CONSUMERS", 1))

#
# -> POST REQUEST with what to fetch
# -> Trigger DAG -> Fetch data from API -> Store to Mongo in RAW collection
# -> Get latest from mongo -> process it -> validate it -> store it in processed + S3


"""
1. POST request to airflow dag with the subject, num_pages, last_page, num_items_per_page
2. Trigger the DAG
3. Fetch data from the API
4. Store the data in the RAW collection in MongoDB
5. Process the data
6. Validate the data
7. Store the data in the PROCESSED collection in MongoDB
8. Store the data in S3
"""

# Unpack the parameters from the DAG run configuration


def unpack_trigger_request(**kwargs):
    subject = kwargs["dag_run"].conf.get("subject", "animals")
    num_pages = kwargs["dag_run"].conf.get("num_pages", 1)
    last_page = kwargs["dag_run"].conf.get("last_page", None)
    items_per_page = kwargs["dag_run"].conf.get("num_items_per_page", 10)

    # Create a dictionary to hold the unpacked parameters
    params = {
        "subject": subject,
        "num_pages": num_pages,
        "last_page": last_page,
        "items_per_page": items_per_page,
    }

    return params


def use_unpacked_parameters(**kwargs):
    # Pull the parameters from XCom
    params = kwargs["ti"].xcom_pull(task_ids="unpack_task")  # 'unpack_task' is the ID of the unpacking task

    # Access the individual parameters
    subject = params["subject"]
    num_pages = params["num_pages"]
    # ... use the parameters in this task ...
    print(f"Subject (from unpacked params): {subject}")
    print(f"Number of Pages (from unpacked params): {num_pages}")

    # Example: Pass parameters to another task using XCom push
    # kwargs['ti'].xcom_push(key='api_params', value=params)  # Explicit push if needed


# Define default arguments for the DAG
default_args = {
    "owner": "your_name",
    "depends_on_past": False,
    "email": ["your_email@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="image_data_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 7, 4),
    schedule_interval=None,
    catchup=False,
) as dag:

    def run_producers(**kwargs):
        """Runs the Unsplash and Pexels producers, pushing to Redis."""
        processors = [
            UnsplashProcessor(
                subject=IMAGES_SUBJECT,
                start_page=START_PAGE,
                num_pages=NUM_PAGES,
                num_items_per_page=NUM_ITEMS_PER_PAGE,
            ),
            PexelsProcessor(
                subject=IMAGES_SUBJECT,
                start_page=START_PAGE,
                num_pages=NUM_PAGES,
                num_items_per_page=NUM_ITEMS_PER_PAGE,
            ),
        ]

        async def run():
            producer_tasks = [asyncio.create_task(processor.run()) for processor in processors]
            await asyncio.gather(*producer_tasks)

        asyncio.run(run())

    fetch_data = PythonOperator(
        task_id="fetch_website_data",
        python_callable=run_producers,
        provide_context=True,
    )

    process_data = PythonOperator(
        task_id="process_data",
        python_callable=run_consumers,
        provide_context=True,
    )

    fetch_data >> process_data
