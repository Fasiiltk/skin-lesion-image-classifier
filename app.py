from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os


import os

# Make sure 'uploads' folder exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")






# Initialize Flask app
app = Flask(__name__)


# Load the trained model
model = load_model("skin_cancer_model.h5")

# Define class labels (same as in the training)
label_mapping = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}

# Function to preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(100, 100))  # Resize image
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:                                                        
        return "No file uploaded!"

    file = request.files['file']
    if file.filename == '':
        return "No file selected!"

    # Save the uploaded file
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = label_mapping[predicted_class]

    return f"Predicted Skin Cancer Type: {predicted_label}"

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
