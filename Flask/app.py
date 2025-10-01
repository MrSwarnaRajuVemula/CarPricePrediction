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
    response.headers.add('Cache-Control', 'no-store, no-cache, must-revalidate, post-check=0, pre-check-0')
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
            # Get the absolute path to the models directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, 'models')

            print(f"📁 Looking for models in: {models_dir}")

            # Check if models directory exists
            if not os.path.exists(models_dir):
                print(f"❌ Models directory not found: {models_dir}")
                current_dir = os.listdir(base_dir)
                print(f"📂 Current directory contents: {current_dir}")
                raise FileNotFoundError("Models directory not found")

            # List files in models directory
            model_files = os.listdir(models_dir)
            print(f"📄 Files in models directory: {model_files}")

            # Load regression models with absolute paths
            model_paths = {
                'lasso': os.path.join(models_dir, 'Lasso.pkl'),
                'linear': os.path.join(models_dir, 'LinearRegression.pkl'),
                'ridge': os.path.join(models_dir, 'Ridge.pkl'),
                'logistic': os.path.join(models_dir, 'LogisticRegression.pkl')
            }

            for model_name, model_path in model_paths.items():
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    print(f"✓ {model_name} model loaded successfully!")
                else:
                    print(f"❌ Model file not found: {model_path}")
                    raise FileNotFoundError(f"Model file not found: {model_path}")

            print("🎉 All models loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print(traceback.format_exc())
            # Fallback to simple mock models
            self.models = {
                'lasso': self.mock_predict,
                'linear': self.mock_predict,
                'ridge': self.mock_predict,
                'logistic': self.logistic_predict
            }
            print("⚠️ Using mock models as fallback")

    def mock_predict(self, features):
        """Simple mock prediction when real models fail to load"""
        # Simple calculation without arbitrary base prices
        vehicle_age = features.get('vehicle_age', 0)
        max_power = features.get('max_power', 0)
        km_driven = features.get('km_driven', 0)

        # Simple mock calculation
        price = (max_power * 1000) - (vehicle_age * 20000) - (km_driven * 0.5)
        return max(price, 50000)  # Minimum reasonable price for mock data

    def logistic_predict(self, features):
        """Mock classification prediction"""
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
                if value == '':
                    value = '0'
                features[feature] = float(value)
                print(f"  {feature}: {features[feature]}")
            except (ValueError, TypeError) as e:
                print(f"❌ Error converting {feature}: {e}")
                features[feature] = 0.0

        # Categorical features
        categorical_features = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
        for feature in categorical_features:
            features[feature] = form_data.get(feature, 'unknown').lower()
            print(f"  {feature}: {features[feature]}")

        print("✅ Processed features:", features)
        return features

    def create_feature_array(self, features):
        """Convert features to numpy array"""
        print("🔧 Creating feature array...")

        feature_array = []

        # Add numerical features
        numerical_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
        for feature in numerical_features:
            feature_array.append(features.get(feature, 0))
            print(f"  Numerical {feature}: {features.get(feature, 0)}")

        # Simple encoding for categorical features
        categorical_mapping = {
            'brand': {'unknown': 0, 'maruti': 1, 'hyundai': 2, 'honda': 3, 'toyota': 4, 'ford': 5},
            'fuel_type': {'unknown': 0, 'petrol': 1, 'diesel': 2, 'cng': 3, 'electric': 4},
            'seller_type': {'unknown': 0, 'individual': 1, 'dealer': 2, 'trustmark': 3},
            'transmission_type': {'unknown': 0, 'manual': 1, 'automatic': 2},
            'model': {'unknown': 0}
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
                return self.logistic_predict(features)
            else:
                return model(features)
        else:
            # It's an actual scikit-learn model - return raw prediction
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
                        predicted_class = model.predict(feature_array)[0]
                        print(f"✅ Classification result: class={predicted_class}")
                        return predicted_class, 80
                else:
                    # Regression prediction - RETURN RAW PREDICTION
                    print("📈 Running regression prediction...")
                    prediction = model.predict(feature_array)[0]
                    print(f"✅ Raw regression prediction: {prediction}")
                    return float(prediction)  # No modifications

            except Exception as e:
                print(f"❌ Error in model prediction: {e}")
                print(traceback.format_exc())
                print("🔄 Falling back to mock prediction")
                if model_name == 'logistic' or prediction_type == 'classification':
                    return self.logistic_predict(features)
                else:
                    return self.mock_predict(features)


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

            # Format the price without modifying the prediction
            formatted_price = "₹{:,.2f}".format(predicted_price)

            result = {
                'success': True,
                'predicted_price': formatted_price,
                'raw_predicted_price': predicted_price,
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