from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model and feature columns
MODEL_PATH = 'gradient_boosting_model.pkl'
FEATURES_PATH = 'feature_columns.pkl'

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Create DataFrame with input data
        input_dict = {}
        for feature in feature_columns:
            if feature in data:
                input_dict[feature] = [data[feature]]
            else:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
        
        input_df = pd.DataFrame(input_dict)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': probability.tolist(),
            'status': 'success'
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get list of required features"""
    return jsonify({
        'features': feature_columns.tolist(),
        'status': 'success'
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
