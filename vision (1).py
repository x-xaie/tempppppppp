import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import requests

# Load the pre-trained MobileNet V2 model from TensorFlow Hub
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
                   output_shape=[1280],
                   trainable=False)
])

# Fetch the class labels for ImageNet
response = requests.get("https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt")
class_labels = np.array(response.text.splitlines())

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize the image to match the input size of MobileNet V2
    image = np.array(image) / 255.0  # Normalize pixel values between 0 and 1
    image = (image - 0.5) * 2.0  # Rescale to [-1, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = predictions[0, predicted_class_index] * 100
    return predicted_class_label, confidence

# Streamlit app
st.title("Image Classification with MobileNet V2")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make predictions when the user clicks the "Classify" button
    if st.button("Classify"):
        # Predict the class label and confidence score
        predicted_label, confidence = predict(image)
        st.write("Prediction:", predicted_label)
        st.write("Confidence:", f"{confidence:.2f}%")
