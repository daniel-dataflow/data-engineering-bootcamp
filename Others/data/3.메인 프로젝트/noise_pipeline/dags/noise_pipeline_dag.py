from airflow.sdk import DAG
from datetime import datetime, timedelta
import sys
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

from src.main import run_pipeline
from src.ml import train_ml_model



with DAG(
    dag_id="noise_data_pipeline_ml",
    schedule="0 3 * * *",
    start_date=datetime(2024, 1, 1),
) as dag:

    fetch_and_send_noise_data = PythonOperator(
        task_id="fetch_and_send_noise_data",
        python_callable=run_pipeline,
    )

    spark_noise_analysis = SparkSubmitOperator(
        task_id="spark_noise_analysis",
        application="/home/big/noise_pipeline/spark/noise_analysis.py",
        conn_id="spark_default",
        verbose=True,
    )

    train_ml = PythonOperator(
        task_id="train_ml_model",
        python_callable=train_ml_model,
    )

    fetch_and_send_noise_data >> spark_noise_analysis >> train_ml
