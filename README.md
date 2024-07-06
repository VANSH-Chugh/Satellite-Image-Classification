# Flask Image Classification

![Flask Image Classification](https://your-image-link.com/banner.png)

This repository contains a Flask web application for image classification using a pre-trained Keras model. Users can upload images, and the application will predict the class of the uploaded image, displaying the result along with the probabilities for each class.

## Features

- **Image Upload**: Users can upload images through a web interface.
- **Image Classification**: The application uses a pre-trained Keras model to classify the uploaded images.
- **Result Display**: The predicted class and the probabilities for all classes are displayed.
- **User-Friendly Interface**: Simple and intuitive web interface.

## Table of Contents

- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Training](#model-training)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)

## Demo

![Demo GIF](https://your-image-link.com/demo.gif)

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

