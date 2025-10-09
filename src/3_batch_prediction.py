import mlflow
import pandas as pd

def predict_churn(customer_data_path, model_name, output_path):
    """Score customers for churn risk"""
    
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:8080")
    # Load model
    model = mlflow.sklearn.load_model(f"models:/{model_name}@champion") # type: ignore
    
    # Load data
    customers = pd.read_csv(customer_data_path)
    
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
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    return results

# Run it
predictions = predict_churn(
    r'..\data\raw\New_Customers.csv',
    'Telco Churn',
    r'..\data\out\New_Customers_Churn_Predictions.csv'
)