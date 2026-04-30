from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_engineer',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Định nghĩa DAG
with DAG(
    dag_id='healthcare_etl_pipeline',
    default_args=default_args,
    start_date=datetime(2026, 4, 1),
    schedule='@daily',
    catchup=False,
    tags=['healthcare', 'etl', 'devsecops'],
) as dag:

    # Task 1: Extract
    extract_task = BashOperator(
        task_id='extract_medical_records',
        bash_command='docker run --rm --network healthcare_mlops_demo_default healthcare_worker python extractors/medical_extractor.py'
    )

    # Task 2: Transform
    transform_task = BashOperator(
        task_id='transform_and_anonymize',
        bash_command='docker run --rm --network healthcare_mlops_demo_default healthcare_worker python transformers/medical_transformer.py'
    )

    # Task 3: Huấn luyện PyTorch + Versioning bằng DVC (ĐÃ FIX LỖI PATHSPEC)
    train_task = BashOperator(
        task_id='train_and_log_model',
        bash_command="""
        docker run --rm --network healthcare_mlops_demo_default \
        -e AWS_ACCESS_KEY_ID=admin \
        -e AWS_SECRET_ACCESS_KEY=SuperSecret123! \
        -v /Users/air/healthcare_mlops_demo:/app \
        healthcare_worker bash -c "pip install pathspec==0.11.2 && python trainers/medical_trainer.py"
        """
    )

    # Nối luồng logic hoàn chỉnh: Extract -> Transform -> Train
    extract_task >> transform_task >> train_task