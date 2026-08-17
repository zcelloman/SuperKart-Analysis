from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('superkart_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df_input = pd.DataFrame([data])
    
    # Feature engineering
    df_input['Price_Per_Weight'] = df_input['Product_MRP'] / df_input['Product_Weight']
    df_input['MRP_x_Allocated_Area'] = df_input['Product_MRP'] * df_input['Product_Allocated_Area']
    
    df_encoded = pd.get_dummies(df_input)
    df_encoded = df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
    
    prediction = model.predict(df_encoded)[0]
    return jsonify({'predicted_sales': float(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
