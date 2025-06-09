from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

model = joblib.load('models/model.joblib')
scaler = joblib.load('models/scaler.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    data_input = request.json
    array = pd.DataFrame([{
        'N': data_input['N'],
        'P': data_input['P'],
        'K': data_input['K'],
        'temperature': data_input['temperature'],
        'humidity': data_input['humidity'],
        'ph': data_input['ph'],
        'rainfall': data_input['rainfall']
    }])

    scaled_data = scaler.transform(array)
    prediction = model.predict(scaled_data)

    prediction_proba = model.predict_proba(scaled_data)
    confidence = np.max(prediction_proba)

    crop_response = {
        'prediction': int(prediction[0]),
        'confidence': float(confidence),  
        'labelEncoder': label_encoder.inverse_transform([int(prediction[0])])[0]
    }

    print('Response:', crop_response)
    return jsonify(crop_response)

model_padi = load_model('models/model.h5')

class_labels = [
    "Bacterial leaf blight",
    "Brown Spot",
    "Leaf Smut"
]

@app.route('/predict_padi', methods=['POST'])
def predict_padi():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    img = Image.open(file.stream)
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model_padi.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = class_labels[np.argmax(predictions)]

    padi_response = {
        'predicted_class': predicted_class,
        'confidence': float(confidence)
    }

    print('Response:', padi_response)
    return jsonify(padi_response)

if __name__ == '__main__':
    app.run(debug=True)