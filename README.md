# 🌍 Integrating_MLOps_in_Travel

> Advanced machine learning platform for travel data analytics, predictive pricing, and customer profiling. Built with production-grade MLOps infrastructure.

**Key Capabilities:** Flight Price Prediction • Gender Classification • Hotel Recommendations • Real-time API • Interactive Dashboard • Automated CI/CD

---

## 📚 Table of Contents

- [System Overview](#system-overview)
- [Installation Guide](#installation-guide)
- [Running Locally](#running-locally)
- [Containerization](#containerization)
- [Kubernetes Orchestration](#kubernetes-orchestration)
- [Data Pipeline (Airflow)](#data-pipeline-airflow)
- [CI/CD Integration (Jenkins)](#cicd-integration-jenkins)
- [Experiment Tracking (MLflow)](#experiment-tracking-mlflow)
- [API Reference](#api-reference)
- [Model Performance](#model-performance)
- [Architecture Diagram](#architecture-diagram)

---

## 🎯 System Overview

### Purpose

This capstone project demonstrates an end-to-end machine learning operations (MLOps) system for the travel industry. The platform handles:

1. **Data Processing** - ETL pipelines for flight, hotel, and user data
2. **Model Development** - Regression and classification models with experiment tracking
3. **Service Deployment** - REST API and interactive dashboard
4. **Infrastructure** - Docker containerization and Kubernetes orchestration
5. **Automation** - Apache Airflow DAGs and Jenkins CI/CD pipelines
6. **Monitoring** - MLflow experiment tracking and model registry

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy, Scikit-learn |
| APIs & Dashboards | Flask, Streamlit |
| ML Experimentation | MLflow 2.9+ |
| Orchestration | Apache Airflow 2.7+ |
| Containerization | Docker, Docker Compose |
| Container Orchestration | Kubernetes, Minikube |
| CI/CD Pipeline | Jenkins LTS |
| Version Control | Git |

---

## 📁 Project Organization

```
travel-intelligence-ml/
│
├── data/
│   ├── flights.csv              # 271,888 flight transaction records
│   ├── hotels.csv               # 40,552 accommodation booking logs
│   └── users.csv                # 1,340 traveler profile records
│
├── models/                      # Serialized ML artifacts (.pkl)
│   ├── flight_price_model.pkl
│   ├── gender_classifier.pkl
│   ├── gender_scaler.pkl
│   ├── le_*.pkl                 # Label encoders
│   ├── regression_meta.json     # Model metadata
│   └── classification_meta.json
│
├── flask_api/                   # REST API Service
│   └── app.py                   # Flask application (2 endpoints)
│
├── streamlit_app/              # Web Dashboard
│   └── app.py                   # Streamlit UI (5 pages)
│
├── mlflow_tracking/            # Experiment Management
│   └── mlflow_train.py          # Multi-run experiment script
│
├── airflow/                     # Workflow Orchestration
│   └── dags/
│       └── travel_pipeline_dag.py  # Daily automated pipeline
│
├── jenkins/                     # CI/CD Configuration
│   └── Jenkinsfile              # Build and deploy pipeline
│
├── kubernetes/                  # Container Orchestration
│   └── deployment.yml           # K8s manifests (Deployment, Service, HPA)
│
├── docker/                      # Container Definitions
│   ├── Dockerfile               # API container image
│   ├── Dockerfile.streamlit     # Dashboard container image
│   └── docker-compose.yml       # Multi-service orchestration
│
├── tests/                       # Unit Tests
│   └── test_api.py              # Pytest suite (8 tests)
│
├── notebooks/                   # Analysis & Prototyping
│   └── travel_mlops_colab.py    # Complete EDA + model development
│
├── docs/                        # Documentation & Assets
│   └── eda_visualizations.png   # Exploratory data analysis plots
│
├── requirements.txt             # Python dependencies
└── README.md
```

---

## ⚙️ Installation Guide

### Prerequisites

**System Requirements:**
- Ubuntu 20.04+ / macOS 11+ / Windows 10+
- Python 3.10 or higher
- 8GB RAM minimum
- 20GB free disk space

**Software Prerequisites:**
```bash
# Required tools
Python 3.10+
pip / conda package managers
Docker Desktop (for containerization)
Minikube (for local Kubernetes)
Git (version control)
curl / Postman (API testing)
```

### Setup Steps

#### 1. Clone Repository
```bash
git clone https://github.com/yourorg/travel-intelligence-ml.git
cd travel-intelligence-ml
```

#### 2. Create Python Environment
```bash
python3 -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### 4. Verify Installation
```bash
python -c "import pandas, sklearn, flask, streamlit, mlflow; print('✅ All packages installed')"
```

---

## 🚀 Running Locally

### Phase 1: Train Models

```bash
# Option A: Run complete notebook (Recommended for first-time)
python notebooks/travel_mlops_colab.py

# Option B: Run MLflow experiments with tracking
python mlflow_tracking/mlflow_train.py

# Verify models were created
ls -lh models/*.pkl
```

**Expected Output:**
```
✅ Models saved to models/ directory
✅ flight_price_model.pkl (17 MB)
✅ gender_classifier.pkl (1.2 MB)
✅ Encoders and scalers (100+ KB each)
```

### Phase 2: Start API Service

```bash
# Terminal 1: Start Flask API
python flask_api/app.py

# Output:
# WARNING: This is a development server. Do not use it in production.
# Running on http://0.0.0.0:5000
```

### Phase 3: Test API Endpoints

```bash
# Health check
curl -s http://localhost:5000/health | python -m json.tool

# Predict flight price
curl -X POST http://localhost:5000/predict/flight-price \
  -H "Content-Type: application/json" \
  -d '{
    "from": "Sao Paulo (SP)",
    "to": "Rio de Janeiro (RJ)",
    "flightType": "economic",
    "time": 2.5,
    "distance": 400,
    "agency": "LATAM",
    "month": 6,
    "dayofweek": 3
  }'

# Predict customer profile
curl -X POST http://localhost:5000/predict/gender \
  -H "Content-Type: application/json" \
  -d '{
    "age": 32,
    "from": "Brasilia (DF)",
    "to": "Florianopolis (SC)",
    "flightType": "premium",
    "price": 800.0,
    "time": 3.0,
    "distance": 900,
    "agency": "GOL",
    "month": 8
  }'
```

### Phase 4: Launch Dashboard

```bash
# Terminal 2: Start Streamlit app
streamlit run streamlit_app/app.py

# Opens at http://localhost:8501
# Features:
# - Analytics Hub with 6 charts
# - Flight Price Predictor
# - Hotel Recommendation Engine
# - Model Performance Metrics
# - Market Analysis Dashboard
```

---

## 🐳 Containerization

### Build Container Images

```bash
# Build API container
docker build -t travel-api:latest -f docker/Dockerfile .

# Build Streamlit container
docker build -t travel-dashboard:latest -f docker/Dockerfile.streamlit .

# Verify images
docker images | grep travel
```

### Run Containers Individually

```bash
# API container
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  travel-api:latest

# Streamlit container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  travel-dashboard:latest
```

### Deploy with Docker Compose

```bash
# Start all services
cd docker
docker-compose up -d

# Verify services are running
docker-compose ps

# View logs
docker-compose logs -f api       # API logs
docker-compose logs -f dashboard # Dashboard logs

# Stop all services
docker-compose down
```

### Service URLs After Docker Compose
- **API:** http://localhost:5000
- **Dashboard:** http://localhost:8501
- **Health Check:** http://localhost:5000/health

---

## ☸️ Kubernetes Orchestration

### Prerequisites
```bash
# Install Minikube
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
```

### Deploy to Kubernetes

```bash
# Start Minikube cluster
minikube start --cpus=4 --memory=8192

# Verify cluster
kubectl cluster-info
kubectl get nodes

# Load Docker images into Minikube
minikube image load travel-api:latest
minikube image load travel-dashboard:latest

# Deploy to Kubernetes
kubectl apply -f kubernetes/deployment.yml

# Monitor deployment
kubectl get deployments -n travel-mlops
kubectl get pods -n travel-mlops
kubectl get svc -n travel-mlops

# Check pod logs
kubectl logs -n travel-mlops deployment/travel-api

# Forward ports for local access
kubectl port-forward -n travel-mlops svc/travel-api-service 5000:5000
kubectl port-forward -n travel-mlops svc/travel-dashboard-service 8501:8501

# Scale deployments
kubectl scale deployment travel-api --replicas=5 -n travel-mlops

# Monitor Horizontal Pod Autoscaling
kubectl get hpa -n travel-mlops -w

# Access Kubernetes Dashboard
minikube dashboard
```

---

## 🔄 Data Pipeline (Airflow)

### Installation & Setup

```bash
# Install Airflow
pip install apache-airflow==2.7.3

# Initialize Airflow database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --password admin123 \
    --firstname MLOps \
    --lastname Admin \
    --role Admin \
    --email admin@company.com

# Set project environment variable
export TRAVEL_PROJECT_PATH="$(pwd)"

# Copy DAG file
cp airflow/dags/travel_pipeline_dag.py ~/airflow/dags/

# Verify DAG is recognized
airflow dags list | grep travel_mlops
```

### Run Airflow Services

```bash
# Terminal 1: Start webserver
airflow webserver --port 8080

# Terminal 2: Start scheduler
airflow scheduler

# Access UI
# Open http://localhost:8080
# Login: admin / admin123
```

### Manage DAG Execution

```bash
# View DAG details
airflow dags info travel_mlops_automated_pipeline

# Trigger manual DAG run
airflow dags trigger travel_mlops_automated_pipeline

# View run history
airflow dags list-runs --dag-id travel_mlops_automated_pipeline

# View task logs
airflow tasks logs travel_mlops_automated_pipeline preprocess_data_task 2024-01-15
```

### DAG Schedule & Tasks

| Task | Operation | Duration |
|------|-----------|----------|
| preprocess_data_task | Load and validate CSVs | 2-3 min |
| feature_engineering_task | Create encoded features | 3-5 min |
| train_regression_task | Train price model | 5-10 min |
| train_classification_task | Train gender model | 5-10 min |
| run_tests_task | Execute unit tests | 2 min |
| validate_artifacts_task | Verify model files | 1 min |
| notify_success_task | Send completion alert | < 1 sec |

**Schedule:** Daily at 2:00 AM UTC

---

## 🔧 CI/CD Integration (Jenkins)

### Jenkins Installation

```bash
# Run Jenkins in Docker
docker run -d -p 8080:8080 -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  jenkins/jenkins:lts-jdk17

# Get initial admin password
docker exec $(docker ps -q -f ancestor=jenkins/jenkins:lts-jdk17) \
  cat /var/jenkins_home/secrets/initialAdminPassword

# Open http://localhost:8080
# Complete setup wizard and install recommended plugins
```

### Configure Pipeline Job

```bash
# 1. Click "New Item"
# 2. Enter job name: "travel-mlops-pipeline"
# 3. Select "Pipeline"
# 4. Under Pipeline section:
#    - Definition: Pipeline script from SCM
#    - SCM: Git
#    - Repository URL: your-github-url
#    - Script Path: jenkins/Jenkinsfile
# 5. Click "Build Now"
```

### Pipeline Stages

| Stage | Actions |
|-------|---------|
| Checkout | Clone repository from Git |
| Dependencies | Install Python packages |
| Tests | Run pytest suite |
| Train | Execute model training |
| Build | Create Docker images |
| Push | Upload to registry |
| Deploy | Update Kubernetes cluster |
| Verify | Health checks on deployed services |

---

## 📊 Experiment Tracking (MLflow)

### Run Experiments

```bash
# Execute training with MLflow tracking
python mlflow_tracking/mlflow_train.py

# Output shows:
# Experiment: Flight_Price_Prediction
#   ├── Run: LinearRegression_Baseline
#   ├── Run: RandomForest_n50
#   ├── Run: RandomForest_n100_BEST
#   └── Run: GradientBoosting_n100
#
# Experiment: Gender_Classification
#   ├── Run: LogisticRegression_Baseline
#   ├── Run: RandomForest_n50
#   └── Run: RandomForest_n100_BEST
```

### Launch MLflow UI

```bash
# Start MLflow tracking server
mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlruns.db

# Open http://localhost:5000

# Features:
# ✅ Compare experiment runs
# ✅ View metrics evolution
# ✅ Download model artifacts
# ✅ Register production models
```

---

## 📡 API Reference

### Base URL
```
http://localhost:5000  # Development
https://api.company.com  # Production
```

### Endpoints

#### 1. Health Check
```
GET /health

Response (200):
{
  "status": "operational",
  "models_available": true,
  "service": "travel_mlops_api"
}
```

#### 2. Root Information
```
GET /

Response (200):
{
  "service": "Travel MLOps REST API",
  "version": "1.0",
  "endpoints": { ... }
}
```

#### 3. Flight Price Prediction
```
POST /predict/flight-price

Request Body:
{
  "from": "Sao Paulo (SP)",
  "to": "Rio de Janeiro (RJ)",
  "flightType": "economic|premium|firstClass",
  "time": 2.5,
  "distance": 400,
  "agency": "LATAM|GOL|Azul|...",
  "month": 1-12,
  "dayofweek": 0-6
}

Response (200):
{
  "prediction": 450.75,
  "currency": "USD",
  "input_data": { ... }
}

Error (400):
{
  "error": "Missing required fields: ['month', 'dayofweek']"
}
```

#### 4. Gender Classification
```
POST /predict/gender

Request Body:
{
  "age": 32,
  "from": "Brasilia (DF)",
  "to": "Florianopolis (SC)",
  "flightType": "premium",
  "price": 800.0,
  "time": 3.0,
  "distance": 900,
  "agency": "GOL",
  "month": 8
}

Response (200):
{
  "prediction": "female",
  "confidence": 72.45,
  "class_probabilities": {
    "female": 72.45,
    "male": 27.55
  },
  "input_data": { ... }
}
```

#### 5. Model Metadata
```
GET /metadata/regression
GET /metadata/classification

Response (200):
{
  "metrics": {
    "r2": 1.0000,
    "rmse": 0.01,
    "mae": 0.00
  },
  "from_cities": [...],
  "to_cities": [...],
  ...
}
```

---

## 📈 Model Performance

### Regression Model (Flight Price)

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Regressor (100 trees) |
| R² Score | 1.0000 |
| Root Mean Squared Error | $0.01 |
| Mean Absolute Error | $0.00 |
| Training Samples | 217,504 |
| Test Samples | 54,384 |

**Features Used:**
- Origin city (encoded)
- Destination city (encoded)
- Flight type (encoded)
- Flight duration (hours)
- Distance (kilometers)
- Airline (encoded)
- Month of travel (1-12)
- Day of week (0-6)

### Classification Model (Gender)

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Classifier (100 trees) |
| Overall Accuracy | 58.6% |
| Weighted Precision | 0.59 |
| Weighted Recall | 0.59 |
| F1 Score | 0.59 |
| Classes | 2 (male, female) |

**Note:** Gender prediction is inherently challenging with demographic data. Current accuracy reflects real-world complexity and class imbalance in the dataset.

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                              │
│  ┌──────────────┬──────────────┬──────────────────────────┐  │
│  │ flights.csv  │  hotels.csv  │ users.csv                │  │
│  │ (271K rows)  │  (40K rows)  │ (1.3K rows)              │  │
│  └──────────────┴──────────────┴──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ⬇
┌─────────────────────────────────────────────────────────────┐
│              PROCESSING PIPELINE (Airflow DAG)               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Data Loading → Feature Engineering → Model Training  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ⬇
┌──────────────────────────┬──────────────────────────┐
│   REGRESSION MODEL       │  CLASSIFICATION MODEL     │
│ (Price Prediction)       │ (Gender Profile)          │
│ R² = 1.0000              │ Accuracy = 58.6%          │
└──────────────────────────┴──────────────────────────┘
         ⬇                           ⬇
┌──────────────────────────────────────────────────────┐
│          MODEL REGISTRY & ARTIFACT STORAGE            │
│        (MLflow + Local File System)                  │
└──────────────────────────────────────────────────────┘
         ⬇                           ⬇
┌──────────────────────────────────────────────────────┐
│         REST API (Flask) + Dashboard (Streamlit)     │
└──────────────────────────────────────────────────────┘
         ⬇                           ⬇
    [Docker]  ────────────────────  [Docker]
         ⬇                           ⬇
┌────────────────────────────────────────────────────┐
│         KUBERNETES ORCHESTRATION                   │
│  API Pod (×3) + Dashboard Pod (2–10 HPA)          │
└────────────────────────────────────────────────────┘
         ⬇
    [Jenkins] ─────→ [Git Push Trigger] ─────→ [CI/CD]
```

---

## 📝 Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_api.py::TestFlightPricePrediction -v

# With coverage report
pytest tests/ --cov=flask_api --cov-report=html

# Expected output:
# test_api.py::TestHealthEndpoints::test_health_check_endpoint PASSED
# test_api.py::TestFlightPricePrediction::test_valid_flight_prediction PASSED
# ... (8 tests total)
# ======================= 8 passed in 2.34s =======================
```

---

## 🔍 Troubleshooting

### Issue: Models not found
```bash
# Solution: Train models first
python notebooks/travel_mlops_colab.py
ls -la models/
```

### Issue: Flask port already in use
```bash
# Solution: Kill process on port 5000
lsof -i :5000
kill -9 <PID>

# Or use different port
python flask_api/app.py --port 5001
```

### Issue: Kubernetes pod crashes
```bash
# Check pod logs
kubectl logs -n travel-mlops <pod-name>

# Describe pod for events
kubectl describe pod -n travel-mlops <pod-name>

# Check resource requests vs available
kubectl top nodes
kubectl top pods -n travel-mlops
```

---

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/name`
2. Commit changes: `git commit -m 'Add feature'`
3. Push branch: `git push origin feature/name`
4. Open pull request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👥 Team

- **Data Science:** Model development and experimentation
- **MLOps:** Pipeline automation and deployment
- **DevOps:** Infrastructure and containerization

---

**Last Updated:** January 2025  
**Version:** 2.0.0  
**Maintained By:** ML Engineering Team
