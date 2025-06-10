from flask import Flask, request, jsonify
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

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
    app.run(host='0.0.0.0', port=5000)
