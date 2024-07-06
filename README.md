# Flask Image Classification


This repository contains a Flask web application for image classification using a pre-trained Keras model. Users can upload images, and the application will predict the class of the uploaded image, displaying the result along with the probabilities for each class.

## Features

- **Image Upload**: Users can upload images through a web interface.
- **Image Classification**: The application uses a pre-trained Keras model to classify the uploaded images.
- **Result Display**: The predicted class and the probabilities for all classes are displayed.
- **User-Friendly Interface**: Simple and intuitive web interface.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Training](#model-training)
- [Folder Structure](#folder-structure)


## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/flask-image-classification.git
    cd flask-image-classification
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python3 -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Ensure the `static/uploads` directory exists**:
    ```sh
    mkdir -p static/uploads
    ```

5. **Download your pre-trained Keras model** and place it in the root directory of the project:
    ```sh
    mv path_to_your_model/my_model.h5 .
    ```

## Usage

1. **Run the Flask application**:
    ```sh
    python app.py
    ```

2. **Open your web browser** and navigate to `http://127.0.0.1:5000/`.

3. **Upload an image** and see the classification result.

## How It Works

### 1. Flask Application Setup

The Flask application is set up to handle HTTP requests. It uses the following routes:
- **GET /**: Renders the upload form.
- **POST /**: Handles the file upload and image classification.

### 2. Image Upload and Storage

When a user uploads an image, the file is saved in the `static/uploads` directory. The `UPLOAD_FOLDER` configuration ensures that the directory exists.

### 3. Image Preprocessing

The uploaded image is preprocessed to match the input requirements of the model. This includes:
- Loading the image.
- Resizing it to the target size (256x256).
- Converting the image to an array and expanding its dimensions to fit the model's expected input shape.

```python
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
```

### 4. Model Prediction

The preprocessed image is fed into the pre-trained Keras model to get predictions. The class with the highest probability is chosen as the predicted class.

python
Copy code
```
def predict_class(img_array):
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_idx]
    return predicted_class, predictions[0]

```

![Image Alt Text](https://github.com/yourusername/yourrepository/blob/main/path/to/image.jpg)

### 5. Result Display
The application renders the result on the same page, showing the uploaded image, the predicted class, and the probabilities for all classes. The HTML template (index.html) handles the display.

### 6. Static File Handling
A separate route is defined to serve the uploaded images from the static/uploads directory.

python
Copy code
```
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
```
Model Training
If you want to train the model yourself, follow these steps:

#### 1. Dataset Preparation
Prepare your dataset of images categorized into classes (e.g., cloudy, desert, green_area, water). Organize the images into the following directories:

train/: Training images for model training.
val/: Validation images for model evaluation during training.
test/: Testing images for final model evaluation.
#### 2. Data Augmentation and Preprocessing
Use ImageDataGenerator from Keras to augment and preprocess your images. Augmentation helps in generating variations of images to improve model robustness.

#### 3. Model Definition
```
Define your convolutional neural network (CNN) model using Keras. Below is a sample model definition:

python
Copy code
model = models.Sequential([
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(4, activation='softmax')
])

```
#### 4. Model Compilation
Compile your model with an optimizer, loss function, and metrics:

python
Copy code
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#### 5. Model Training
Train your model using the prepared dataset:

python
Copy code
history = model.fit(
    train_generator,
    epochs=4,
    validation_data=val_generator
)
#### 6. Model Evaluation
Evaluate your model on the test dataset to measure its performance:

python
Copy code
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")
#### 7. Save the Trained Model
Save your trained model for later use:

python
Copy code
model.save('my_model.h5')
## Folder Structure
php
Copy code
flask-image-classification/
```
│
├── static/
│   ├── uploads/           # Directory for storing uploaded images
│   └── style.css          # (Optional) Add your custom styles
│
├── templates/
│   └── index.html         # HTML template for the web interface
│
├── app.py                 # Main Flask application
├── my_model.h5            # Pre-trained Keras model
├── requirements.txt       # List of required packages
└── README.md              # This README file
```

