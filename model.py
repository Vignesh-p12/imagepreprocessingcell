# model.py
import tensorflow as tf  # type: ignore
from tensorflow.keras.applications import VGG16  # type: ignore
from tensorflow.keras.layers import Dense, Flatten  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
import numpy as np  # type: ignore

def load_model():
    # Load the VGG16 model, excluding the top fully connected layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Add custom layers on top of the base model
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)  # Add a dense layer with ReLU activation
    x = Dense(4, activation='softmax')(x)  # Output layer for 4 classes (Benign, Early, Pre, Pro)
    
    # Create a new model
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def predict_image(model, preprocessed_image):
    # Make a prediction with the model
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]
    return predicted_class, confidence
# test_import.py
from model import load_model, predict_image

print("Imports successful!")
print("Loading model module...")

# Load the model to verify if it works without errors
model = load_model()
print("Model loaded successfully!")
