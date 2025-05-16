# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model and encoders
model = joblib.load('liver_disease_model.pkl')
le_dict = joblib.load('liver_label_encoders.pkl')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Column mapping to match your training data
COLUMN_MAPPING = {
    'Age': 'Age',
    'Gender': 'Gender',
    'Total Bilirubin': 'Total Bilirubin',
    'Direct Bilirubin': 'Direct Bilirubin',
    'Alkaline Phosphotase': 'Alkaline Phosphotase',
    'Sgpt': 'Sgpt',
    'Sgot': 'Sgot',
    'Total Proteins': 'Total Proteins',
    'Albumin_G': 'Albumin_G',
    'A/G Ratio': 'A/G Ratio'
}

def preprocess_data(input_data):
    """Preprocess the input data to match training format"""
    # Create a DataFrame with consistent columns
    df = pd.DataFrame([input_data])
    
    # Rename columns to match training data
    df.rename(columns=COLUMN_MAPPING, inplace=True)
    
    # Handle missing/ambiguous values
    ambiguous_values = ['?', 'None', 'none', 'Not Mentioned', 'not mentioned', 
                       'N/A', '', 'Unknown', 'unknown', 'No', 'no']
    df.replace(ambiguous_values, np.nan, inplace=True)
    
    # Convert to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    # Encode categorical variables using saved encoders
    for col in df.select_dtypes(include='object').columns:
        if col in le_dict:
            le = le_dict[col]
            df[col] = le.transform(df[col])
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for single prediction"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Preprocess the input data
        processed_data = preprocess_data(data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[:, 1][0]
        
        # Prepare response
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'interpretation': 'Liver Disease' if prediction[0] == 1 else 'No Liver Disease'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """Endpoint for batch predictions from CSV"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('/tmp', filename)
            file.save(filepath)
            
            # Read and preprocess CSV
            df = pd.read_csv(filepath)
            df.rename(columns=COLUMN_MAPPING, inplace=True)
            
            # Preprocess data
            for col in df.select_dtypes(include='object').columns:
                if col in le_dict:
                    le = le_dict[col]
                    df[col] = le.transform(df[col])
            
            # Make predictions
            predictions = model.predict(df)
            probabilities = model.predict_proba(df)[:, 1]
            
            # Prepare results
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results.append({
                    'patient_id': i,
                    'prediction': int(pred),
                    'probability': float(prob),
                    'interpretation': 'Liver Disease' if pred == 1 else 'No Liver Disease'
                })
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({'results': results})
        
        else:
            return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return """
    <h1>Liver Disease Prediction API</h1>
    <p>Endpoints:</p>
    <ul>
        <li><strong>POST /predict</strong> - Single prediction (JSON input)</li>
        <li><strong>POST /predict_csv</strong> - Batch predictions (CSV file upload)</li>
    </ul>
    """

if __name__ == '__main__':
    app.run(debug=True)