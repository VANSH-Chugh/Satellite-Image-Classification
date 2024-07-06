import os
from flask import Flask, flash, request, redirect, url_for, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Define the path to your trained model
MODEL_PATH = 'my_model.h5'
# Define target size for model input
target_size = (256, 256)

# Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)
# Load class labels from the generator
class_labels = ['cloudy', 'desert', 'green_area', 'water']  # Replace with your actual class labels

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.secret_key = "secret key"

# Function to preprocess image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict class
def predict_class(img_array):
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_idx]
    return predicted_class, predictions[0]

# Route to upload page
@app.route('/')
def upload_form():
    return render_template('index.html')

# Route to handle file upload and classification
@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        flash('Image successfully uploaded and classified')
        # Preprocess the uploaded image
        img_array = preprocess_image(filepath)
        # Predict the class of the uploaded image
        predicted_class, probabilities = predict_class(img_array)
        # Render the classification result page
        return render_template('index.html', filename=filename, predicted_class=predicted_class, probabilities=probabilities)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
