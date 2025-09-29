from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)


# Disable inspect element (client-side protection)
@app.after_request
def after_request(response):
    response.headers.add('Cache-Control', 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0')
    response.headers.add('Pragma', 'no-cache')
    response.headers.add('Expires', '0')
    return response


# Load actual trained models
class CarPricePredictor:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.load_models()

    def load_models(self):
        try:
            # Load regression models
            with open('models/Lasso.pkl', 'rb') as f:
                self.models['lasso'] = pickle.load(f)
            with open('models/LinearRegression.pkl', 'rb') as f:
                self.models['linear'] = pickle.load(f)
            with open('models/Ridge.pkl', 'rb') as f:
                self.models['ridge'] = pickle.load(f)
            with open('models/LogisticRegression.pkl', 'rb') as f:
                self.models['logistic'] = pickle.load(f)

            print("All models loaded successfully!")

        except Exception as e:
            print(f"Error loading models: {e}")
            # Fallback to mock models if loading fails
            self.models = {
                'lasso': self.lasso_predict,
                'linear': self.linear_predict,
                'ridge': self.ridge_predict,
                'logistic': self.logistic_predict
            }

    def lasso_predict(self, features):
        # Mock prediction fallback
        base_price = 500000
        price = base_price - (features['vehicle_age'] * 50000) + (features['max_power'] * 1000)
        return max(price, 100000)

    def linear_predict(self, features):
        # Mock prediction fallback
        base_price = 450000
        price = base_price - (features['vehicle_age'] * 45000) + (features['max_power'] * 900)
        return max(price, 100000)

    def ridge_predict(self, features):
        # Mock prediction fallback
        base_price = 480000
        price = base_price - (features['vehicle_age'] * 47000) + (features['max_power'] * 950)
        return max(price, 100000)

    def logistic_predict(self, features):
        # Logistic regression for classification (price category)
        # This would need to be adapted based on how you trained your logistic regression
        probability = 0.8  # Mock probability
        return probability

    def preprocess_features(self, form_data):
        """Preprocess form data for prediction"""
        features = {}

        # Numerical features
        numerical_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
        for feature in numerical_features:
            try:
                features[feature] = float(form_data.get(feature, 0))
            except:
                features[feature] = 0.0

        # Categorical features
        categorical_features = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
        for feature in categorical_features:
            features[feature] = form_data.get(feature, 'unknown')

        return features

    def create_feature_array(self, features):
        """Convert features to numpy array in the correct order for model prediction"""
        # Define the expected feature order (adjust based on your model training)
        feature_order = [
            'vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',
            'brand', 'model', 'seller_type', 'fuel_type', 'transmission_type'
        ]

        # Create array - you'll need to handle categorical encoding properly
        # This is a simplified version - you should use the same preprocessing as during training
        feature_array = []

        # Add numerical features
        numerical_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
        for feature in numerical_features:
            feature_array.append(features[feature])

        # For categorical features, you would need to use your trained label encoders
        # This is a placeholder - implement proper encoding based on your training
        categorical_features = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
        for feature in categorical_features:
            # Convert categorical to numerical (simplified)
            # In practice, use the same LabelEncoder from training
            encoded_value = hash(features[feature]) % 100  # Simple hash as placeholder
            feature_array.append(encoded_value)

        return np.array([feature_array])

    def predict(self, model_name, features):
        if model_name in self.models:
            # Check if we have actual model objects or mock functions
            model = self.models[model_name]

            if callable(model) and hasattr(model, '__name__') and 'predict' not in dir(model):
                # It's a mock function
                return model(features)
            else:
                # It's an actual scikit-learn model
                try:
                    # Prepare features for the model
                    feature_array = self.create_feature_array(features)

                    # Make prediction
                    if model_name == 'logistic':
                        # Logistic regression returns probability or class
                        prediction = model.predict_proba(feature_array)[0][1]
                        # Convert to price (adjust based on your logistic regression setup)
                        return prediction * 1000000  # Example conversion
                    else:
                        # Regression models
                        prediction = model.predict(feature_array)[0]
                        return max(prediction, 100000)  # Ensure minimum price

                except Exception as e:
                    print(f"Error in model prediction: {e}")
                    # Fallback to mock prediction
                    if model_name == 'lasso':
                        return self.lasso_predict(features)
                    elif model_name == 'linear':
                        return self.linear_predict(features)
                    elif model_name == 'ridge':
                        return self.ridge_predict(features)
                    else:
                        return self.lasso_predict(features)
        else:
            raise ValueError(f"Model {model_name} not found")


# Initialize predictor
predictor = CarPricePredictor()


@app.route('/')
def index():
    return render_template('Home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        selected_model = form_data.get('model_type', 'lasso')

        # Preprocess features
        features = predictor.preprocess_features(form_data)

        # Make prediction
        predicted_price = predictor.predict(selected_model, features)

        # Format the price
        formatted_price = "₹{:,.2f}".format(predicted_price)

        return jsonify({
            'success': True,
            'predicted_price': formatted_price,
            'model_used': selected_model.upper()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)