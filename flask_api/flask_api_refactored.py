"""
Travel Recommendation System - REST API Service
Handles flight price estimation and customer profiling predictions
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify

# Application initialization
application = Flask(__name__)

# Configuration paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

# Load all required machine learning models
class ModelManager:
    """Centralized model loading and management"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.regression_model = joblib.load(os.path.join(model_path, 'flight_price_model.pkl'))
        self.from_encoder = joblib.load(os.path.join(model_path, 'le_from.pkl'))
        self.to_encoder = joblib.load(os.path.join(model_path, 'le_to.pkl'))
        self.flight_type_encoder = joblib.load(os.path.join(model_path, 'le_flighttype.pkl'))
        self.agency_encoder = joblib.load(os.path.join(model_path, 'le_agency.pkl'))
        
        self.classification_model = joblib.load(os.path.join(model_path, 'gender_classifier.pkl'))
        self.feature_scaler = joblib.load(os.path.join(model_path, 'gender_scaler.pkl'))
        self.gender_encoder = joblib.load(os.path.join(model_path, 'le_gender.pkl'))
        self.clf_from_encoder = joblib.load(os.path.join(model_path, 'clf_le_from.pkl'))
        self.clf_to_encoder = joblib.load(os.path.join(model_path, 'clf_le_to.pkl'))
        self.clf_flight_type_encoder = joblib.load(os.path.join(model_path, 'clf_le_flighttype.pkl'))
        self.clf_agency_encoder = joblib.load(os.path.join(model_path, 'clf_le_agency.pkl'))
        
        with open(os.path.join(model_path, 'regression_meta.json')) as metadata:
            self.regression_metadata = json.load(metadata)
        with open(os.path.join(model_path, 'classification_meta.json')) as metadata:
            self.classification_metadata = json.load(metadata)

model_manager = ModelManager(MODEL_DIR)

def encode_value_safely(encoder, input_value):
    """
    Safely encode categorical values using a pre-trained encoder.
    Returns the first class if the value is not found.
    
    Args:
        encoder: Fitted LabelEncoder instance
        input_value: Value to encode
        
    Returns:
        int: Encoded value
    """
    available_classes = list(encoder.classes_)
    if input_value in available_classes:
        return encoder.transform([input_value])[0]
    return 0


@application.route('/', methods=['GET'])
def root_handler():
    """Provides API documentation and available endpoints"""
    return jsonify({
        "service": "Travel MLOps REST API",
        "version": "1.0",
        "endpoints": {
            "health_check": "GET /health",
            "flight_price": "POST /predict/flight-price",
            "customer_profile": "POST /predict/gender",
            "regression_info": "GET /metadata/regression",
            "classification_info": "GET /metadata/classification"
        }
    })

@application.route('/health', methods=['GET'])
def health_check():
    """Service health status endpoint"""
    return jsonify({
        "status": "operational",
        "models_available": True,
        "service": "travel_mlops_api"
    }), 200

@application.route('/metadata/regression', methods=['GET'])
def regression_metadata():
    """Regression model metadata and configuration"""
    return jsonify(model_manager.regression_metadata)

@application.route('/metadata/classification', methods=['GET'])
def classification_metadata():
    """Classification model metadata and configuration"""
    return jsonify(model_manager.classification_metadata)

@application.route('/predict/flight-price', methods=['POST'])
def estimate_flight_price():
    """
    Estimate ticket price based on flight parameters.
    
    Expected JSON payload:
    {
        "from": "origin_city",
        "to": "destination_city", 
        "flightType": "firstClass|premium|economic",
        "time": flight_duration_hours,
        "distance": distance_km,
        "agency": "agency_name",
        "month": month_number,
        "dayofweek": day_number
    }
    """
    try:
        payload = request.get_json(force=True)
        
        required_fields = ['from', 'to', 'flightType', 'time', 'distance', 
                          'agency', 'month', 'dayofweek']
        missing_fields = [field for field in required_fields if field not in payload]
        
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

        input_features = np.array([[
            encode_value_safely(model_manager.from_encoder, payload['from']),
            encode_value_safely(model_manager.to_encoder, payload['to']),
            encode_value_safely(model_manager.flight_type_encoder, payload['flightType']),
            float(payload['time']),
            float(payload['distance']),
            encode_value_safely(model_manager.agency_encoder, payload['agency']),
            int(payload['month']),
            int(payload['dayofweek'])
        ]])

        predicted_price = model_manager.regression_model.predict(input_features)[0]

        return jsonify({
            "prediction": round(float(predicted_price), 2),
            "currency": "USD",
            "input_data": payload
        })

    except Exception as error:
        return jsonify({"error": str(error)}), 500


@application.route('/predict/gender', methods=['POST'])
def predict_customer_profile():
    """
    Predict customer gender/profile based on travel behavior.
    
    Expected JSON payload:
    {
        "age": customer_age,
        "from": "origin_city",
        "to": "destination_city",
        "flightType": "firstClass|premium|economic",
        "price": ticket_price,
        "time": flight_duration,
        "distance": distance_km,
        "agency": "agency_name",
        "month": month_number
    }
    """
    try:
        payload = request.get_json(force=True)
        
        required_fields = ['age', 'from', 'to', 'flightType', 'price', 
                          'time', 'distance', 'agency', 'month']
        missing_fields = [field for field in required_fields if field not in payload]
        
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

        input_features = np.array([[
            float(payload['age']),
            encode_value_safely(model_manager.clf_from_encoder, payload['from']),
            encode_value_safely(model_manager.clf_to_encoder, payload['to']),
            encode_value_safely(model_manager.clf_flight_type_encoder, payload['flightType']),
            float(payload['price']),
            float(payload['time']),
            float(payload['distance']),
            encode_value_safely(model_manager.clf_agency_encoder, payload['agency']),
            int(payload['month'])
        ]])

        scaled_features = model_manager.feature_scaler.transform(input_features)
        predicted_class_encoded = model_manager.classification_model.predict(scaled_features)[0]
        prediction_probabilities = model_manager.classification_model.predict_proba(scaled_features)[0]
        
        predicted_class = model_manager.gender_encoder.inverse_transform([predicted_class_encoded])[0]
        confidence_score = round(float(max(prediction_probabilities)) * 100, 2)

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence_score,
            "class_probabilities": {
                cls: round(float(prob) * 100, 2)
                for cls, prob in zip(model_manager.gender_encoder.classes_, prediction_probabilities)
            },
            "input_data": payload
        })

    except Exception as error:
        return jsonify({"error": str(error)}), 500


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
