# 📞 Telco Customer Churn Prediction

An end-to-end **machine learning project** that predicts telecom customer churn using **Logistic Regression**, **Random Forest**, and **XGBoost**, tracked with **MLflow** and deployed via **FastAPI**.

This project demonstrates a **production-style workflow** including exploratory data analysis (EDA), model training with experiment tracking, batch prediction, and deployment as an API — following modern MLOps practices.

---

## 🧭 Project Overview

Customer churn prediction helps telecom companies **identify customers likely to leave** and take proactive retention actions.  
In this project, I built, compared, and deployed several ML models to predict whether a customer will churn based on their service usage and demographic information.

**Key Objectives:**
- Analyze customer data through EDA.
- Build and evaluate multiple models (Logistic Regression, Random Forest, XGBoost).
- Track experiments with **MLflow** for reproducibility.
- Serve predictions through a **FastAPI** endpoint.

---

## 🧠 Data Description

**Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)

- **Rows:** ~7,000 customers  
- **Target:** `Churn` (Yes / No)  
- **Features:**  
  - Customer demographics  
  - Account information (tenure, contract, payment method)  
  - Service usage (internet, phone, TV)  

---

## ⚙️ Tech Stack

| Category | Tools |
|-----------|-------|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling | scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| Deployment | FastAPI |
| Environment | venv |

---

## 🚀 Workflow

### 1. Exploratory Data Analysis (EDA)
- Reviewed data quality
- Investigated variables' descriptive statistics.
- Generated `EDA_report.html` for summary insights.

### 2. Model Training & Evaluation
- Compared three models: Logistic Regression, Random Forest, and XGBoost.  
- Used train-test split, cross-validation, hyperparameter tuning, and AUC as the main metric.  
- Logged all experiments, metrics, and parameters with **MLflow**.

#### 🧾 MLflow Tracking
Each training run is logged to MLflow, capturing:
- Model type and hyperparameters
- Training and validation metrics
- Versioned model artifacts
- Chart artifacts

To start the MLflow UI:
```
mlflow localhost:8765
```

Then open: http://localhost:8765


| Model | Test AUC |
|-------|-----|
| Logistic Regression | 0.8607 |
| Random Forest | 0.8642 |
| XGBoost | 0.8641 |

### 3. Batch Prediction
- Implemented the batch prediction for scoring new customer data using AirFlow.  
- Outputs predictions in CSV for further business action.

To start AirFlow:
```
airflow standalone
```

Then open: http://localhost:8080/


### 4. Deployment
- Exposed the final Random Forest model as a REST API using **FastAPI** (`4_deployment.py`).  
- Endpoint:
  - POST /predict
  - Content-Type: application/json

Example request:
```json
{"tenure": 12, "MonthlyCharges": 70.35, "Contract": "Month-to-month", ...}
```

To start FastAPI:
```
python3 ./src/4_deployment.py
```
Then open: http://localhost:8000/


## 🧰 Setup Instructions
1. Clone the Repository

```
git clone https://github.com/<username>/telco-churn-prediction.git

cd telco-churn-prediction
```

2. Create Environment
Using venv:
```
python -m venv venv
source venv/bin/activate     # (Linux/Mac)
venv\Scripts\activate        # (Windows)
pip install -r requirements-full.txt
```
3. Run Training Script
Open the notebooks and run sequencially.

4. Launch FastAPI
```
python notebooks/4_deployment.py
```

## 📈 Future Improvements
Add Streamlit dashboard for business-friendly visualization

Log model drift and performance over time

This project demonstrates practical machine learning engineering and MLOps practices for customer churn prediction. Built as part of a personal portfolio to showcase full-cycle data science capabilities.

