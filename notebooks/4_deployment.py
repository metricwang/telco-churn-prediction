"""
FastAPI deployment for Telco Churn Prediction Model

Run:
uvicorn fastapi_churn_api:app --reload

Test:
http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================
# Set up MLflow
mlflow.set_tracking_uri("http://localhost:8080")

# Set the best model's ID from MLflow
MODEL_NAME = f"Telco Churn"
MODEL_ALIAS = f"champion"
MODEL_PATH = f"models:/{MODEL_NAME}@champion"

client = mlflow.tracking.MlflowClient() # type: ignore

# Resolve alias to model version
model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

# Now load threshold from tags
CHURN_THRESHOLD = float(model_version.tags.get("optimal_threshold", 0.5))

print(f"Loaded model: {MODEL_NAME}@{MODEL_ALIAS}")
print(f"Version: {model_version.version}")
print(f"Threshold: {CHURN_THRESHOLD}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# LOAD MODEL
# ============================================================================

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predict customer churn probability using ML model",
    version="1.0.0"
)

# Load model once at startup (not on every request)
try:
    model = mlflow.sklearn.load_model(MODEL_PATH) # type: ignore
    logger.info(f"Model loaded successfully from {MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class CustomerFeatures(BaseModel):
    """Input features for a single customer"""
    
    # Numerical features
    tenure: int = Field(..., ge=0, description="Number of months with company")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in dollars")
    TotalCharges: float = Field(..., ge=0, description="Total charges in dollars")
    
    # Categorical features
    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 or 1")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(..., description="Payment method type")
    
    class Config:
        schema_extra = {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 70.35,
                "TotalCharges": 844.20,
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check"
            }
        }


class BatchCustomers(BaseModel):
    """Input for batch predictions"""
    customers: List[CustomerFeatures]


class ChurnPrediction(BaseModel):
    """Prediction output for a single customer"""
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    will_churn: bool = Field(..., description="Predicted churn (based on threshold)")
    risk_level: str = Field(..., description="Low, Medium, or High risk")
    confidence: str = Field(..., description="Model confidence level")


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[ChurnPrediction]
    total_customers: int
    high_risk_count: int
    threshold_used: float


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_risk_level(probability: float) -> str:
    """Categorize churn probability into risk levels"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


def calculate_confidence(probability: float) -> str:
    """Assess model confidence based on probability distance from 0.5"""
    distance_from_uncertain = abs(probability - 0.5)
    
    if distance_from_uncertain > 0.3:
        return "High"
    elif distance_from_uncertain > 0.15:
        return "Medium"
    else:
        return "Low"


def customer_to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    """Convert Pydantic model to DataFrame for prediction"""
    return pd.DataFrame([customer.model_dump()])


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "message": "Telco Churn Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.post("/predict", response_model=ChurnPrediction)
def predict_single_customer(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer
    
    Returns:
        ChurnPrediction with probability, binary prediction, risk level, and confidence
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        customer_df = customer_to_dataframe(customer)
        
        # Predict
        probability = float(model.predict_proba(customer_df)[0, 1])
        will_churn = bool(probability >= CHURN_THRESHOLD)
        risk_level = calculate_risk_level(probability)
        confidence = calculate_confidence(probability)
        
        logger.info(f"Prediction: prob={probability:.3f}, churn={will_churn}")
        
        return ChurnPrediction(
            churn_probability=round(probability, 4),
            will_churn=will_churn,
            risk_level=risk_level,
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_customers(batch: BatchCustomers):
    """
    Predict churn probability for multiple customers
    
    Returns:
        List of predictions with summary statistics
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert all customers to DataFrame
        customers_df = pd.DataFrame([c.dict() for c in batch.customers])
        
        # Predict
        probabilities = model.predict_proba(customers_df)[:, 1]
        
        # Create predictions
        predictions = []
        high_risk_count = 0
        
        for prob in probabilities:
            will_churn = bool(prob >= CHURN_THRESHOLD)
            risk_level = calculate_risk_level(prob)
            
            if risk_level == "High":
                high_risk_count += 1
            
            predictions.append(ChurnPrediction(
                churn_probability=round(float(prob), 4),
                will_churn=will_churn,
                risk_level=risk_level,
                confidence=calculate_confidence(prob)
            ))
        
        logger.info(f"Batch prediction: {len(predictions)} customers, {high_risk_count} high risk")
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            high_risk_count=high_risk_count,
            threshold_used=CHURN_THRESHOLD
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")



# ============================================================================
# TESTING ENDPOINTS (Development only - remove in production)
# ============================================================================

@app.post("/test/high-risk")
def test_high_risk_customer():
    """Test endpoint with a high-risk customer profile"""
    
    test_customer = CustomerFeatures(
        tenure=1,
        MonthlyCharges=90.0,
        TotalCharges=90.0,
        gender="Male",
        SeniorCitizen=0,
        Partner="No",
        Dependents="No",
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="Yes",
        StreamingMovies="Yes",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check"
    )
    
    return predict_single_customer(test_customer)


@app.post("/test/low-risk")
def test_low_risk_customer():
    """Test endpoint with a low-risk customer profile"""
    
    test_customer = CustomerFeatures(
        tenure=60,
        MonthlyCharges=50.0,
        TotalCharges=3000.0,
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="Yes",
        PhoneService="Yes",
        MultipleLines="Yes",
        InternetService="DSL",
        OnlineSecurity="Yes",
        OnlineBackup="Yes",
        DeviceProtection="Yes",
        TechSupport="Yes",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Two year",
        PaperlessBilling="No",
        PaymentMethod="Bank transfer (automatic)"
    )
    
    return predict_single_customer(test_customer)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)