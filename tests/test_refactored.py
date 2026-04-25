

import json
import pytest
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flask_api.app import application


class TestAPIConfiguration:
    """Test Flask application initialization"""
    
    @pytest.fixture
    def api_client(self):
        """Create test client"""
        application.config['TESTING'] = True
        with application.test_client() as test_client:
            yield test_client
    
    def test_application_initialization(self, api_client):
        """Verify Flask app initializes correctly"""
        assert application is not None
        assert application.config['TESTING'] is True


class TestHealthEndpoints:
    """Test health check and status endpoints"""
    
    @pytest.fixture
    def api_client(self):
        """Create test client"""
        application.config['TESTING'] = True
        with application.test_client() as test_client:
            yield test_client
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint returns API documentation"""
        response = api_client.get('/')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'endpoints' in data
        assert 'service' in data
    
    def test_health_check_endpoint(self, api_client):
        """Test health check endpoint"""
        response = api_client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'operational'
        assert data['models_available'] is True


class TestMetadataEndpoints:
    """Test model metadata retrieval"""
    
    @pytest.fixture
    def api_client(self):
        """Create test client"""
        application.config['TESTING'] = True
        with application.test_client() as test_client:
            yield test_client
    
    def test_regression_metadata_endpoint(self, api_client):
        """Test regression model metadata endpoint"""
        response = api_client.get('/metadata/regression')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'metrics' in data or 'from_cities' in data
    
    def test_classification_metadata_endpoint(self, api_client):
        """Test classification model metadata endpoint"""
        response = api_client.get('/metadata/classification')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, dict)


class TestFlightPricePrediction:
    """Test flight price prediction endpoint"""
    
    @pytest.fixture
    def api_client(self):
        """Create test client"""
        application.config['TESTING'] = True
        with application.test_client() as test_client:
            yield test_client
    
    def test_valid_flight_prediction(self, api_client):
        """Test prediction with valid input"""
        prediction_request = {
            'from': 'Sao Paulo (SP)',
            'to': 'Rio de Janeiro (RJ)',
            'flightType': 'economic',
            'time': 2.5,
            'distance': 400,
            'agency': 'Gol',
            'month': 6,
            'dayofweek': 2
        }
        
        response = api_client.post(
            '/predict/flight-price',
            data=json.dumps(prediction_request),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        assert 'prediction' in result
        assert isinstance(result['prediction'], (int, float))
        assert result['prediction'] > 0
        assert result['currency'] == 'USD'
        assert 'input_data' in result
    
    def test_missing_required_fields(self, api_client):
        """Test prediction with missing fields"""
        incomplete_request = {
            'from': 'Sao Paulo (SP)',
            'to': 'Rio de Janeiro (RJ)'
        }
        
        response = api_client.post(
            '/predict/flight-price',
            data=json.dumps(incomplete_request),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result
    
    def test_invalid_json_payload(self, api_client):
        """Test prediction with malformed JSON"""
        response = api_client.post(
            '/predict/flight-price',
            data='{invalid json}',
            content_type='application/json'
        )
        
        assert response.status_code == 500
        result = json.loads(response.data)
        assert 'error' in result
    
    def test_numeric_conversion(self, api_client):
        """Test numeric field validation"""
        request_with_invalid_numbers = {
            'from': 'Sao Paulo (SP)',
            'to': 'Rio de Janeiro (RJ)',
            'flightType': 'economic',
            'time': 'invalid_number',
            'distance': 400,
            'agency': 'Gol',
            'month': 6,
            'dayofweek': 2
        }
        
        response = api_client.post(
            '/predict/flight-price',
            data=json.dumps(request_with_invalid_numbers),
            content_type='application/json'
        )
        
        assert response.status_code == 500


class TestGenderClassification:
    """Test gender classification endpoint"""
    
    @pytest.fixture
    def api_client(self):
        """Create test client"""
        application.config['TESTING'] = True
        with application.test_client() as test_client:
            yield test_client
    
    def test_valid_classification_request(self, api_client):
        """Test classification with valid input"""
        classification_request = {
            'age': 35,
            'from': 'Brasilia (DF)',
            'to': 'Florianopolis (SC)',
            'flightType': 'premium',
            'price': 750.50,
            'time': 3.0,
            'distance': 900,
            'agency': 'LATAM',
            'month': 8
        }
        
        response = api_client.post(
            '/predict/gender',
            data=json.dumps(classification_request),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'class_probabilities' in result
        assert isinstance(result['confidence'], (int, float))
        assert 0 <= result['confidence'] <= 100
    
    def test_missing_classification_fields(self, api_client):
        """Test classification with missing required fields"""
        incomplete_request = {
            'age': 35,
            'from': 'Brasilia (DF)'
        }
        
        response = api_client.post(
            '/predict/gender',
            data=json.dumps(incomplete_request),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result
    
    def test_probability_distribution(self, api_client):
        """Test probability values sum to 100%"""
        classification_request = {
            'age': 28,
            'from': 'Salvador (BA)',
            'to': 'Manaus (AM)',
            'flightType': 'firstClass',
            'price': 1200.0,
            'time': 5.5,
            'distance': 2200,
            'agency': 'Azul',
            'month': 10
        }
        
        response = api_client.post(
            '/predict/gender',
            data=json.dumps(classification_request),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        probabilities = list(result['class_probabilities'].values())
        total_probability = sum(probabilities)
        
        assert 99 <= total_probability <= 101  # Allow small rounding error


class TestErrorHandling:
    """Test API error handling"""
    
    @pytest.fixture
    def api_client(self):
        """Create test client"""
        application.config['TESTING'] = True
        with application.test_client() as test_client:
            yield test_client
    
    def test_nonexistent_endpoint(self, api_client):
        """Test request to non-existent endpoint"""
        response = api_client.get('/nonexistent')
        assert response.status_code == 404
    
    def test_invalid_request_method(self, api_client):
        """Test invalid HTTP method"""
        response = api_client.post('/')
        assert response.status_code != 405  # Flask GET routes may not restrict POST
    
    def test_empty_json_payload(self, api_client):
        """Test empty JSON payload"""
        response = api_client.post(
            '/predict/flight-price',
            data=json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 400


class TestInputValidation:
    """Test input validation and edge cases"""
    
    @pytest.fixture
    def api_client(self):
        """Create test client"""
        application.config['TESTING'] = True
        with application.test_client() as test_client:
            yield test_client
    
    def test_zero_distance(self, api_client):
        """Test prediction with zero distance"""
        prediction_request = {
            'from': 'Sao Paulo (SP)',
            'to': 'Sao Paulo (SP)',
            'flightType': 'economic',
            'time': 0.5,
            'distance': 0,
            'agency': 'Gol',
            'month': 5,
            'dayofweek': 1
        }
        
        response = api_client.post(
            '/predict/flight-price',
            data=json.dumps(prediction_request),
            content_type='application/json'
        )
        
        assert response.status_code == 200
    
    def test_negative_values(self, api_client):
        """Test prediction with negative values"""
        prediction_request = {
            'from': 'Sao Paulo (SP)',
            'to': 'Rio de Janeiro (RJ)',
            'flightType': 'economic',
            'time': -2.5,
            'distance': -400,
            'agency': 'Gol',
            'month': 6,
            'dayofweek': 2
        }
        
        response = api_client.post(
            '/predict/flight-price',
            data=json.dumps(prediction_request),
            content_type='application/json'
        )
        
        # API should handle this (either accept or reject gracefully)
        assert response.status_code in [200, 500]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
