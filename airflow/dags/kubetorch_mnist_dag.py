from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from kubetorch_example.tasks import data_preprocessing, deploy_inference, run_training


default_args = {
    "owner": "matt",
    "depends_on_past": False,
    "start_date": datetime(2024, 8, 6),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    "schedule_interval": None,  # or your desired schedule
    "catchup": False,
    "max_active_runs": 1,
}

with DAG(
    "mnist_training_pipeline",
    default_args=default_args,
    description="A simple PyTorch training DAG with multiple steps",
    schedule=None,
) as dag:

    preprocess_data_task = PythonOperator(
        task_id="preprocess_data_task",
        python_callable=data_preprocessing,
        dag=dag,
    )

    train_model_task = PythonOperator(
        task_id="train_model_task",
        python_callable=run_training,
        dag=dag,
    )

    deploy_inference_task = PythonOperator(
        task_id="deploy_inference_task",
        python_callable=deploy_inference,
        dag=dag,
    )

    preprocess_data_task >> train_model_task >> deploy_inference_task
