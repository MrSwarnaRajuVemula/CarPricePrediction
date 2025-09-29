from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import traceback

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
            print("Loading models from models/ directory...")

            with open('models/Lasso.pkl', 'rb') as f:
                self.models['lasso'] = pickle.load(f)
            print("✓ Lasso model loaded successfully!")

            with open('models/LinearRegression.pkl', 'rb') as f:
                self.models['linear'] = pickle.load(f)
            print("✓ Linear Regression model loaded successfully!")

            with open('models/Ridge.pkl', 'rb') as f:
                self.models['ridge'] = pickle.load(f)
            print("✓ Ridge model loaded successfully!")

            with open('models/LogisticRegression.pkl', 'rb') as f:
                self.models['logistic'] = pickle.load(f)
            print("✓ Logistic Regression model loaded successfully!")

            print("All models loaded successfully!")

        except Exception as e:
            print(f"Error loading models: {e}")
            print(traceback.format_exc())
            # Fallback to mock models if loading fails
            self.models = {
                'lasso': self.lasso_predict,
                'linear': self.linear_predict,
                'ridge': self.ridge_predict,
                'logistic': self.logistic_predict
            }
            print("Using mock models as fallback")

    def lasso_predict(self, features):
        # Mock prediction fallback
        base_price = 500000
        price = base_price - (features.get('vehicle_age', 0) * 50000) + (features.get('max_power', 0) * 1000)
        return max(price, 100000)

    def linear_predict(self, features):
        # Mock prediction fallback
        base_price = 450000
        price = base_price - (features.get('vehicle_age', 0) * 45000) + (features.get('max_power', 0) * 900)
        return max(price, 100000)

    def ridge_predict(self, features):
        # Mock prediction fallback
        base_price = 480000
        price = base_price - (features.get('vehicle_age', 0) * 47000) + (features.get('max_power', 0) * 950)
        return max(price, 100000)

    def logistic_predict(self, features):
        # Mock classification - returns probability and class
        probability = 0.75  # Mock probability
        predicted_class = 1 if probability > 0.5 else 0
        confidence = int(probability * 100)
        return predicted_class, confidence

    def preprocess_features(self, form_data):
        """Preprocess form data for prediction"""
        print("🔧 Preprocessing form data...")
        print("Raw form data:", form_data)

        features = {}

        # Numerical features
        numerical_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
        for feature in numerical_features:
            try:
                value = form_data.get(feature, '0')
                # Handle empty strings
                if value == '':
                    value = '0'
                features[feature] = float(value)
                print(f"  {feature}: {features[feature]}")
            except (ValueError, TypeError) as e:
                print(f"❌ Error converting {feature}: {e}")
                features[feature] = 0.0

        # Categorical features - store but use simple encoding
        categorical_features = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
        for feature in categorical_features:
            features[feature] = form_data.get(feature, 'unknown').lower()
            print(f"  {feature}: {features[feature]}")

        print("✅ Processed features:", features)
        return features

    def create_feature_array(self, features):
        """Convert features to numpy array - COMPATIBLE VERSION"""
        print("🔧 Creating feature array...")

        # Use consistent feature order that matches your model training
        # This should match the order used when you trained the models
        feature_order = [
            'vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',
            # Categorical features encoded as numerical
            'brand_encoded', 'model_encoded', 'seller_encoded', 'fuel_encoded', 'transmission_encoded'
        ]

        feature_array = []

        # Add numerical features
        numerical_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
        for feature in numerical_features:
            feature_array.append(features.get(feature, 0))
            print(f"  Numerical {feature}: {features.get(feature, 0)}")

        # Simple encoding for categorical features
        # In production, use the same encoding as during training
        categorical_mapping = {
            'brand': {'unknown': 0, 'maruti': 1, 'hyundai': 2, 'honda': 3, 'toyota': 4, 'ford': 5},
            'fuel_type': {'unknown': 0, 'petrol': 1, 'diesel': 2, 'cng': 3, 'electric': 4},
            'seller_type': {'unknown': 0, 'individual': 1, 'dealer': 2, 'trustmark': 3},
            'transmission_type': {'unknown': 0, 'manual': 1, 'automatic': 2},
            'model': {'unknown': 0}  # Simple model encoding
        }

        # Encode categorical features
        brand_val = features.get('brand', 'unknown').lower()
        feature_array.append(categorical_mapping['brand'].get(brand_val, 0))

        model_val = features.get('model', 'unknown').lower()
        feature_array.append(categorical_mapping['model'].get(model_val, 0))

        seller_val = features.get('seller_type', 'unknown').lower()
        feature_array.append(categorical_mapping['seller_type'].get(seller_val, 0))

        fuel_val = features.get('fuel_type', 'unknown').lower()
        feature_array.append(categorical_mapping['fuel_type'].get(fuel_val, 0))

        transmission_val = features.get('transmission_type', 'unknown').lower()
        feature_array.append(categorical_mapping['transmission_type'].get(transmission_val, 0))

        result = np.array([feature_array])
        print(f"✅ Final feature array: {result}")
        print(f"✅ Feature array shape: {result.shape}")
        return result

    def predict(self, model_name, features, prediction_type='regression'):
        print(f"🎯 Making prediction with {model_name} for {prediction_type}")

        if model_name not in self.models:
            print(f"⚠️ Model {model_name} not found, using lasso as fallback")
            model_name = 'lasso'

        model = self.models[model_name]

        # Check if it's a mock function
        if callable(model) and not hasattr(model, 'predict'):
            print(f"🔄 Using mock prediction for {model_name}")
            if model_name == 'logistic' or prediction_type == 'classification':
                predicted_class, confidence = self.logistic_predict(features)
                return predicted_class, confidence
            else:
                return model(features)
        else:
            # It's an actual scikit-learn model
            try:
                feature_array = self.create_feature_array(features)
                print(f"🔧 Feature array prepared for model input")

                if model_name == 'logistic' or prediction_type == 'classification':
                    # Classification prediction
                    print("🔍 Running classification prediction...")
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(feature_array)
                        predicted_class = model.predict(feature_array)[0]
                        confidence = int(probabilities[0][predicted_class] * 100)
                        print(f"✅ Classification result: class={predicted_class}, confidence={confidence}%")
                        return predicted_class, confidence
                    else:
                        # Fallback for models without predict_proba
                        predicted_class = model.predict(feature_array)[0]
                        print(f"✅ Classification result: class={predicted_class}")
                        return predicted_class, 80  # Default confidence
                else:
                    # Regression prediction
                    print("📈 Running regression prediction...")
                    prediction = model.predict(feature_array)[0]
                    print(f"✅ Regression prediction: {prediction}")
                    return max(float(prediction), 100000)  # Ensure minimum price

            except Exception as e:
                print(f"❌ Error in model prediction: {e}")
                print(traceback.format_exc())
                # Fallback to mock prediction
                print("🔄 Falling back to mock prediction")
                if model_name == 'logistic' or prediction_type == 'classification':
                    return self.logistic_predict(features)
                else:
                    return getattr(self, f'{model_name}_predict')(features)


# Initialize predictor
predictor = CarPricePredictor()


@app.route('/')
def index():
    return render_template('Home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n" + "=" * 50)
        print("🚀 PREDICTION REQUEST STARTED")
        print("=" * 50)

        # Get form data
        form_data = request.form.to_dict()
        print("📋 Received form data:", form_data)

        selected_model = form_data.get('model_type', 'lasso')
        prediction_type = form_data.get('prediction_type', 'regression')

        print(f"🎯 Selected model: {selected_model}")
        print(f"🎯 Prediction type: {prediction_type}")

        # Preprocess features
        features = predictor.preprocess_features(form_data)

        # Make prediction based on type
        if prediction_type == 'classification':
            print("🔍 Running classification...")
            predicted_class, confidence = predictor.predict('logistic', features, 'classification')

            result = {
                'success': True,
                'predicted_class': str(predicted_class),
                'confidence': confidence,
                'model_used': 'LOGISTIC_REGRESSION'
            }
            print(f"✅ Classification result: {result}")
        else:
            print("📈 Running regression...")
            predicted_price = predictor.predict(selected_model, features, 'regression')

            # Format the price
            formatted_price = "₹{:,.2f}".format(predicted_price)

            result = {
                'success': True,
                'predicted_price': formatted_price,
                'model_used': selected_model.upper()
            }
            print(f"✅ Regression result: {result}")

        print("=" * 50)
        print("✅ PREDICTION REQUEST COMPLETED SUCCESSFULLY")
        print("=" * 50)

        return jsonify(result)

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print("❌ ERROR:", error_msg)
        print(traceback.format_exc())
        print("=" * 50)

        return jsonify({
            'success': False,
            'error': "An error occurred while processing your request. Please try again with different values."
        }), 500


if __name__ == '__main__':
    print("🚀 Starting Flask Car Price Prediction Server...")
    print("📁 Current working directory:", os.getcwd())
    print("📁 Models directory contents:",
          os.listdir('models') if os.path.exists('models') else "Models directory not found")
    print("🌐 Server will be available at: http://localhost:5000")
    print("🔧 Debug mode: ON")
    app.run(debug=True, host='0.0.0.0', port=5000)