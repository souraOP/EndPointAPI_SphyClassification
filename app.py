from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

model = tf.keras.models.load_model('soil.h5')

@app.route('/predict', methods=['POST'])

def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    
    # Read and preprocess the image
    image = Image.open(io.BytesIO(image_file.read()))
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0) 
    
    try:
        prediction = model.predict(image_array)[0]
        
        result = {
            'silt': f"{prediction[0]*100:.2f}%",
            'gravel': f"{prediction[1]*100:.2f}%",
            'sand': f"{prediction[2]*100:.2f}%"
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5051)
