import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# Set the console to use UTF-8 encoding
import sys
import io

# Ensure standard output uses UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)

# Load the trained model
model = load_model('best_model.keras')

# Define class mappings
classes = {
    4: ('nv', ' melanocytic nevi'), 
    6: ('mel', 'melanoma'),
    2: ('bkl', 'benign keratosis-like lesions'), 
    1: ('bcc', ' basal cell carcinoma'),
    5: ('vasc', ' pyogenic granulomas and hemorrhage'), 
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    3: ('df', 'dermatofibroma')
}

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')


def preprocess_image(image_path):
    """Preprocess the uploaded image for prediction."""
    img = load_img(image_path, target_size=(28, 28))  # Resize image to model's input size
    img_array = img_to_array(img) / 255.0  # Convert to array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Preprocess the uploaded image
            preprocessed_image = preprocess_image(file_path)

            # Make prediction
            prediction = model.predict(preprocessed_image)

            # Process prediction results
            max_prob = np.max(prediction[0])
            class_index = np.argmax(prediction[0])
            class_name = classes[class_index]
            
            # Prepare response
            response = {
                'Predicted probabilities': prediction[0].tolist(),  # Convert to a list
                'Predicted class index': int(class_index),  # Convert np.int64 to int
                'class name': class_name,
                'max probability': float(max_prob)  # Convert np.float64 to float
            }
            return jsonify(response)

    # Render the upload form if it's a GET request
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
