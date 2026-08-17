from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import io

app = Flask(__name__)
model = joblib.load('superkart_model.joblib')

def preprocess_features(df):
    """Applies required feature transformations and aligns columns to the fitted model."""
    df_temp = df.copy()
    
    # 1. Feature Engineering: Interaction metrics
    df_temp['Price_Per_Weight'] = df_temp['Product_MRP'] / df_temp['Product_Weight']
    df_temp['Price_Per_Weight'] = df_temp['Price_Per_Weight'].replace([np.inf, -np.inf], np.nan)
    df_temp['Price_Per_Weight'] = df_temp['Price_Per_Weight'].fillna(df_temp['Price_Per_Weight'].median())
    
    df_temp['MRP_x_Allocated_Area'] = df_temp['Product_MRP'] * df_temp['Product_Allocated_Area']
    
    # 2. One-hot encode and reindex to align with model expectations
    df_encoded = pd.get_dummies(df_temp)
    df_encoded = df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
    
    return df_encoded

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df_input = pd.DataFrame([data])
    df_ready = preprocess_features(df_input)
    prediction = model.predict(df_ready)[0]
    return jsonify({'predicted_sales': round(float(prediction), 2)}), 200

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Read the incoming CSV bytes into a DataFrame
    df_batch = pd.read_csv(io.BytesIO(file.read()))
    
    # Process features and run batch inference
    df_ready = preprocess_features(df_batch)
    predictions = model.predict(df_ready)
    
    return jsonify({
        'predictions': [round(float(p), 2) for p in predictions]
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
