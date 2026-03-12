# ✈️ Travel MLOps Capstone Project

> **End-to-end Machine Learning pipeline** for travel data: Flight Price Prediction, Gender Classification, and Hotel Recommendations — deployed with Flask, Docker, Kubernetes, Airflow, Jenkins, and MLflow.

---

## 📁 Project Structure

```
travel-mlops-capstone/
├── data/
│   ├── flights.csv          # 271,888 flight records
│   ├── hotels.csv           # 40,552 hotel bookings
│   └── users.csv            # 1,340 user profiles
│
├── models/                  # Saved ML models (.pkl)
│   ├── flight_price_model.pkl
│   ├── gender_classifier.pkl
│   ├── gender_scaler.pkl
│   ├── le_from.pkl / le_to.pkl / le_flighttype.pkl / le_agency.pkl
│   ├── le_gender.pkl
│   ├── regression_meta.json
│   └── classification_meta.json
│
├── notebooks/
│   └── travel_mlops_colab.py   # Full Colab notebook (EDA + Models)
│
├── flask_api/
│   └── app.py              # REST API (Flight Price + Gender endpoints)
│
├── streamlit_app/
│   └── app.py              # Interactive dashboard + Hotel Recommender
│
├── docker/
│   ├── Dockerfile           # Flask API container
│   ├── Dockerfile.streamlit # Streamlit container
│   └── docker-compose.yml   # Multi-service orchestration
│
├── kubernetes/
│   └── deployment.yml      # K8s Deployments, Services, HPA
│
├── airflow/
│   └── dags/
│       └── travel_pipeline_dag.py  # Automated ML pipeline DAG
│
├── jenkins/
│   └── Jenkinsfile         # CI/CD pipeline definition
│
├── mlflow_tracking/
│   └── mlflow_train.py     # Experiment tracking for all models
│
├── tests/
│   └── test_api.py         # Unit tests for Flask API
│
├── docs/                   # Generated plots & reports
├── requirements.txt
└── README.md
```

---

## 🤖 Models Built

| Model | Algorithm | Target | Key Metric |
|-------|-----------|--------|------------|
| Flight Price Regression | Random Forest Regressor | Flight price (USD) | R² = 1.0000, RMSE = $0.01 |
| Gender Classification | Random Forest Classifier | User gender | Accuracy = 58.6% |
| Hotel Recommendation | Content-Based Filtering | Hotel ranking | Precision@5 |

---

## 🚀 Quick Start — Local Setup

### Prerequisites
```bash
# Required software:
Python 3.10+
Docker Desktop
Minikube (for Kubernetes)
Apache Airflow 2.7+
Jenkins LTS
MLflow 2.9+
```

### Step 1: Clone & Install
```bash
git clone 
cd travel-mlops-capstone
pip install -r requirements.txt
```

### Step 2: Train Models
```bash
python notebooks/travel_mlops_colab.py
# Models saved to models/ directory
```

### Step 3: Run Flask API (Local)
```bash
python flask_api/app.py
# API running at http://localhost:5000
```

### Step 4: Test the API
```bash
# Health check
curl http://localhost:5000/health

# Predict flight price
curl -X POST http://localhost:5000/predict/flight-price \
  -H "Content-Type: application/json" \
  -d '{
    "from": "Recife (PE)",
    "to": "Florianopolis (SC)",
    "flightType": "firstClass",
    "time": 2.5,
    "distance": 700,
    "agency": "FlyingDrops",
    "month": 9,
    "dayofweek": 3
  }'

# Predict gender
curl -X POST http://localhost:5000/predict/gender \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "from": "Recife (PE)",
    "to": "Florianopolis (SC)",
    "flightType": "economic",
    "price": 500.0,
    "time": 2.5,
    "distance": 700,
    "agency": "FlyingDrops",
    "month": 6
  }'
```

### Step 5: Run Streamlit App
```bash
streamlit run streamlit_app/app.py
# Open http://localhost:8501
```

---

## 🐳 Docker Deployment

```bash
# Build and run API container
cd travel-mlops-capstone
docker build -t travel-mlops-api -f docker/Dockerfile .
docker run -p 5000:5000 travel-mlops-api

# Build and run Streamlit container
docker build -t travel-mlops-streamlit -f docker/Dockerfile.streamlit .
docker run -p 8501:8501 travel-mlops-streamlit

# Run both with Docker Compose
docker-compose -f docker/docker-compose.yml up -d

# Verify
docker ps
curl http://localhost:5000/health
```

---

## ☸️ Kubernetes Deployment

```bash
# Start Minikube
minikube start --memory=4096 --cpus=2

# Load Docker images into Minikube
minikube image load travel-mlops-api:latest
minikube image load travel-mlops-streamlit:latest

# Deploy
kubectl apply -f kubernetes/deployment.yml

# Check status
kubectl get pods -n travel-mlops
kubectl get services -n travel-mlops

# Get service URL
minikube service travel-api-service -n travel-mlops --url
minikube service travel-streamlit-service -n travel-mlops --url

# Scale manually
kubectl scale deployment travel-api --replicas=5 -n travel-mlops

# Verify HPA
kubectl get hpa -n travel-mlops
```

---

## 🔄 Apache Airflow Setup

```bash
# Install Airflow
pip install apache-airflow==2.7.3

# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin --password admin \
    --firstname Admin --lastname User \
    --role Admin --email admin@example.com

# Set project path environment variable
export TRAVEL_PROJECT_PATH=/path/to/travel-mlops-capstone

# Copy DAG to Airflow dags folder
cp airflow/dags/travel_pipeline_dag.py ~/airflow/dags/

# Start Airflow
airflow webserver --port 8080 &
airflow scheduler &

# Access UI at http://localhost:8080
# Username: admin / Password: admin
# Enable and trigger: travel_mlops_pipeline DAG
```

---

## 🔧 Jenkins CI/CD Setup

```bash
# 1. Install Jenkins (via Docker for easiest setup)
docker run -d -p 8080:8080 -p 50000:50000 \
    -v jenkins_home:/var/jenkins_home \
    jenkins/jenkins:lts-jdk17

# 2. Get initial admin password
docker exec <container_id> cat /var/jenkins_home/secrets/initialAdminPassword

# 3. Access at http://localhost:8080
# Install suggested plugins

# 4. Create new Pipeline job:
#    - New Item → Pipeline → "travel-mlops-pipeline"
#    - Pipeline Definition → "Pipeline script from SCM"
#    - SCM: Git, Repo URL: your GitHub URL
#    - Script Path: jenkins/Jenkinsfile

# 5. Click "Build Now" to trigger pipeline
# Pipeline stages: Checkout → Install → Test → Train → Docker → Deploy
```

---

## 📊 MLflow Tracking

```bash
# Install MLflow
pip install mlflow==2.9.2

# Run experiments (trains multiple models and logs metrics)
python mlflow_tracking/mlflow_train.py

# Launch MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlruns.db
# Open http://localhost:5000

# View:
# - Experiments: Flight_Price_Prediction, Gender_Classification
# - Runs: LinearRegression, RandomForest_n50, RandomForest_n100, GradientBoosting
# - Metrics: RMSE, R², MAE, Accuracy, Precision, Recall, F1
# - Registered Models: FlightPricePredictor, GenderClassifier
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
# Expected: 8 tests pass
```

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and available endpoints |
| GET | `/health` | Health check |
| GET | `/metadata/regression` | Regression model metadata |
| GET | `/metadata/classification` | Classification model metadata |
| POST | `/predict/flight-price` | Predict flight price |
| POST | `/predict/gender` | Predict user gender |

---

## 📈 Model Results

### Regression (Flight Price)
- **Algorithm**: Random Forest Regressor (100 trees)
- **Features**: origin, destination, flight type, duration, distance, agency, month, day
- **R² Score**: 1.0000
- **RMSE**: $0.01
- **MAE**: $0.00

### Classification (Gender)
- **Algorithm**: Random Forest Classifier (100 trees)
- **Features**: age, flight features, price, travel patterns
- **Accuracy**: 58.6%
- **Precision**: 0.59 | **Recall**: 0.59 | **F1**: 0.59

---

## 🏗️ Architecture Overview

```
[Raw CSVs] → [Feature Engineering] → [Model Training]
                                           ↓
                              [.pkl Model Files saved]
                                           ↓
                    ┌──────────────────────┴──────────────────┐
                    │                                          │
             [Flask REST API]                      [Streamlit Dashboard]
                    │                                          │
             [Docker Container]                     [Docker Container]
                    │                                          │
             [Kubernetes Pod ×3]                   [Kubernetes Pod ×2]
                    │                                    (HPA: 2–10)
                    └──────────────────────┬──────────────────┘
                                           │
                              [Airflow DAG — Daily Schedule]
                              [Jenkins CI/CD — On Git Push]
                              [MLflow — Experiment Tracking]
```

---
