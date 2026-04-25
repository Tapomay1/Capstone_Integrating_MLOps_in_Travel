"""
Automated Travel ML Pipeline - Apache Airflow DAG
Orchestrates daily data processing, model training, and artifact management
Execution: Daily at 2:00 AM UTC
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# Project Configuration
PROJECT_ROOT = os.getenv('TRAVEL_PROJECT_PATH', '/path/to/travel-mlops-capstone')
sys.path.insert(0, PROJECT_ROOT)

# DAG Configuration
default_configuration = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['ops@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=4)
}

# Define DAG
travel_ml_workflow = DAG(
    dag_id='travel_mlops_automated_pipeline',
    default_args=default_configuration,
    description='Complete travel ML training and deployment pipeline',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
    tags=['ml-pipeline', 'travel', 'production']
)


def fetch_and_preprocess_data(**context):
    """
    Load datasets from CSV files and perform initial preprocessing
    """
    import pandas as pd
    
    data_location = os.path.join(PROJECT_ROOT, 'data')
    
    print("[TASK] Loading datasets...")
    
    flight_records = pd.read_csv(os.path.join(data_location, 'flights.csv'))
    hotel_records = pd.read_csv(os.path.join(data_location, 'hotels.csv'))
    user_records = pd.read_csv(os.path.join(data_location, 'users.csv'))
    
    # Data validation
    print(f"[VALIDATE] Flights: {len(flight_records)} records")
    print(f"[VALIDATE] Hotels: {len(hotel_records)} records")
    print(f"[VALIDATE] Users: {len(user_records)} records")
    
    # Store data paths in XCom for downstream tasks
    context['task_instance'].xcom_push(
        key='data_paths',
        value={
            'flights': os.path.join(data_location, 'flights.csv'),
            'hotels': os.path.join(data_location, 'hotels.csv'),
            'users': os.path.join(data_location, 'users.csv')
        }
    )
    
    print("[SUCCESS] Data loading completed")


def execute_feature_engineering(**context):
    """
    Transform raw data into engineered features for modeling
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    
    print("[TASK] Starting feature engineering...")
    
    data_config = context['task_instance'].xcom_pull(
        task_ids='preprocess_data_task',
        key='data_paths'
    )
    
    flight_df = pd.read_csv(data_config['flights'])
    
    # Temporal feature extraction
    flight_df['date'] = pd.to_datetime(flight_df['date'], format='%m/%d/%Y')
    flight_df['travel_month'] = flight_df['date'].dt.month
    flight_df['travel_dayofweek'] = flight_df['date'].dt.dayofweek
    
    # Categorical encoding
    categorical_features = ['from', 'to', 'flightType', 'agency']
    
    for feature_name in categorical_features:
        encoder = LabelEncoder()
        flight_df[f'{feature_name}_encoded'] = encoder.fit_transform(flight_df[feature_name])
        print(f"[ENCODE] {feature_name}: {len(encoder.classes_)} classes")
    
    print("[SUCCESS] Feature engineering completed")
    
    context['task_instance'].xcom_push(
        key='features_ready',
        value=True
    )


def train_regression_model(**context):
    """
    Train and evaluate flight price prediction model
    """
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    print("[TASK] Training regression model...")
    
    data_config = context['task_instance'].xcom_pull(
        task_ids='preprocess_data_task',
        key='data_paths'
    )
    
    flight_df = pd.read_csv(data_config['flights'])
    flight_df['date'] = pd.to_datetime(flight_df['date'], format='%m/%d/%Y')
    flight_df['travel_month'] = flight_df['date'].dt.month
    flight_df['travel_dayofweek'] = flight_df['date'].dt.dayofweek
    
    # Prepare features and target
    feature_columns = ['from', 'to', 'flightType', 'time', 'distance', 'agency', 'travel_month', 'travel_dayofweek']
    
    # Quick encoding for training
    for col in ['from', 'to', 'flightType', 'agency']:
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        flight_df[col + '_enc'] = encoder.fit_transform(flight_df[col])
    
    training_features = flight_df[['from_enc', 'to_enc', 'flightType_enc', 'time', 'distance', 'agency_enc', 'travel_month', 'travel_dayofweek']]
    training_target = flight_df['price']
    
    # Split data
    X_training, X_testing, y_training, y_testing = train_test_split(
        training_features, 
        training_target, 
        test_size=0.2, 
        random_state=42
    )
    
    # Train model
    regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    regressor.fit(X_training, y_training)
    
    # Evaluate
    predictions = regressor.predict(X_testing)
    rmse = np.sqrt(mean_squared_error(y_testing, predictions))
    r2 = r2_score(y_testing, predictions)
    mae = mean_absolute_error(y_testing, predictions)
    
    print(f"[METRICS] R² Score: {r2:.4f}")
    print(f"[METRICS] RMSE: ${rmse:.2f}")
    print(f"[METRICS] MAE: ${mae:.2f}")
    
    # Save model
    model_dir = os.path.join(PROJECT_ROOT, 'models')
    joblib.dump(regressor, os.path.join(model_dir, 'flight_price_model.pkl'))
    
    print("[SUCCESS] Regression model trained and saved")


def train_classification_model(**context):
    """
    Train and evaluate gender classification model
    """
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print("[TASK] Training classification model...")
    
    data_config = context['task_instance'].xcom_pull(
        task_ids='preprocess_data_task',
        key='data_paths'
    )
    
    flight_df = pd.read_csv(data_config['flights'])
    user_df = pd.read_csv(data_config['users'])
    
    flight_df['date'] = pd.to_datetime(flight_df['date'], format='%m/%d/%Y')
    flight_df['travel_month'] = flight_df['date'].dt.month
    
    # Merge datasets
    merged_df = flight_df.merge(user_df, left_on='userCode', right_on='code', how='inner')
    
    # Prepare features
    feature_columns = ['age', 'from', 'to', 'flightType', 'price', 'time', 'distance', 'agency', 'travel_month']
    
    for col in ['from', 'to', 'flightType', 'agency']:
        encoder = LabelEncoder()
        merged_df[col + '_enc'] = encoder.fit_transform(merged_df[col])
    
    training_features = merged_df[['age', 'from_enc', 'to_enc', 'flightType_enc', 'price', 'time', 'distance', 'agency_enc', 'travel_month']]
    training_target = merged_df['gender']
    
    # Encode target
    target_encoder = LabelEncoder()
    training_target_encoded = target_encoder.fit_transform(training_target)
    
    # Split data
    X_training, X_testing, y_training, y_testing = train_test_split(
        training_features,
        training_target_encoded,
        test_size=0.2,
        random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_training_scaled = scaler.fit_transform(X_training)
    X_testing_scaled = scaler.transform(X_testing)
    
    # Train model
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    classifier.fit(X_training_scaled, y_training)
    
    # Evaluate
    predictions = classifier.predict(X_testing_scaled)
    accuracy = accuracy_score(y_testing, predictions)
    precision = precision_score(y_testing, predictions, average='weighted')
    recall = recall_score(y_testing, predictions, average='weighted')
    f1 = f1_score(y_testing, predictions, average='weighted')
    
    print(f"[METRICS] Accuracy: {accuracy:.4f}")
    print(f"[METRICS] Precision: {precision:.4f}")
    print(f"[METRICS] Recall: {recall:.4f}")
    print(f"[METRICS] F1 Score: {f1:.4f}")
    
    # Save model and preprocessing objects
    model_dir = os.path.join(PROJECT_ROOT, 'models')
    joblib.dump(classifier, os.path.join(model_dir, 'gender_classifier.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'gender_scaler.pkl'))
    joblib.dump(target_encoder, os.path.join(model_dir, 'le_gender.pkl'))
    
    print("[SUCCESS] Classification model trained and saved")


def run_api_tests(**context):
    """
    Execute unit tests on the Flask API
    """
    print("[TASK] Running API tests...")
    
    test_dir = os.path.join(PROJECT_ROOT, 'tests')
    
    test_result = os.system(f'cd {PROJECT_ROOT} && python -m pytest {test_dir} -v --tb=short')
    
    if test_result == 0:
        print("[SUCCESS] All tests passed")
    else:
        print("[WARNING] Some tests failed - review logs")


def validate_model_artifacts(**context):
    """
    Verify all required model files exist
    """
    print("[TASK] Validating model artifacts...")
    
    required_models = [
        'flight_price_model.pkl',
        'gender_classifier.pkl',
        'gender_scaler.pkl',
        'le_gender.pkl'
    ]
    
    model_dir = os.path.join(PROJECT_ROOT, 'models')
    
    for model_file in required_models:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"[OK] {model_file} ({file_size:.2f} MB)")
        else:
            print(f"[ERROR] {model_file} missing!")
            raise FileNotFoundError(f"Model {model_file} not found")
    
    print("[SUCCESS] All models validated")


# Define Tasks

preprocess_task = PythonOperator(
    task_id='preprocess_data_task',
    python_callable=fetch_and_preprocess_data,
    dag=travel_ml_workflow
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering_task',
    python_callable=execute_feature_engineering,
    dag=travel_ml_workflow
)

regression_training_task = PythonOperator(
    task_id='train_regression_task',
    python_callable=train_regression_model,
    dag=travel_ml_workflow
)

classification_training_task = PythonOperator(
    task_id='train_classification_task',
    python_callable=train_classification_model,
    dag=travel_ml_workflow
)

testing_task = PythonOperator(
    task_id='run_tests_task',
    python_callable=run_api_tests,
    dag=travel_ml_workflow
)

validation_task = PythonOperator(
    task_id='validate_artifacts_task',
    python_callable=validate_model_artifacts,
    dag=travel_ml_workflow
)

success_notification = BashOperator(
    task_id='notify_success_task',
    bash_command='echo "Pipeline completed successfully at $(date)"',
    dag=travel_ml_workflow
)

# Define Workflow Dependencies
preprocess_task >> feature_engineering_task
feature_engineering_task >> [regression_training_task, classification_training_task]
[regression_training_task, classification_training_task] >> testing_task >> validation_task >> success_notification
