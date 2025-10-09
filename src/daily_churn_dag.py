from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import pandas as pd
import mlflow

# --- Define paths ---
BASE_DIR = '/home/ll/Documents/telco-churn-prediction/'
DATA_DIR = os.path.join(BASE_DIR, "data")
PREDICTION_DIR = os.path.join(DATA_DIR, "predictions")

# --- Define functions ---
def ingest_data():
    """Simulate new daily customer data ingestion."""
    src_file = os.path.join(DATA_DIR, "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    dest_file = os.path.join(DATA_DIR, f"daily_churn_today.csv")
    df = pd.read_csv(src_file)
    # Simulate small daily changes (shuffle or sample)
    df.sample(frac=0.01).to_csv(dest_file, index=False)
    print(f"✅ Ingested data saved to {dest_file}")

def run_batch_prediction():
    """Run the batch prediction script using subprocess."""
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:8765")
    # Load model
    model_name = 'Telco Churn'
    model = mlflow.sklearn.load_model(f"models:/{model_name}@champion") # type: ignore
    
    # Load data
    customers = pd.read_csv(os.path.join(DATA_DIR, f"daily_churn_today.csv"))
    
    # Predict
    probabilities = model.predict_proba(customers)[:, 1]
    
    # Create output
    results = customers.copy()
    results['churn_probability'] = probabilities
    results['risk_level'] = pd.cut(
        probabilities, 
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    # Save
    results.to_csv(os.path.join(DATA_DIR, f"daily_churn_today_pred.csv"), index=False)
    print("✅ Batch prediction completed.")

# --- Define default args ---
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# --- Define the DAG ---
with DAG(
    dag_id="daily_churn_dag",
    default_args=default_args,
    description="Daily pipeline for churn prediction",
    schedule=timedelta(days=1),
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["churn", "ml", "mlflow"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data
    )

    predict_task = PythonOperator(
        task_id="run_batch_prediction",
        python_callable=run_batch_prediction
    )

    # --- Set dependencies ---
    ingest_task >> predict_task
