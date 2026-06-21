# =============================================================================
# app/digit_recognizer.py — CNN Digit Recognition
# Owner: Member 3
# 
# Functions:
# 1. load_model()  : load the trained digit recognition model to be used by predict_grid()
# 2. predict_grid(): using the model to transform 81 cell images (28x28) to 9x9 list of integers 
#		             * integers: 0 = empty, 1-9 = digit
# 3. build_model() : builds a fresh CNN model with the same architecture as train_model.py,
#                    included here for reference and re-training purposes
# =============================================================================

# Import libraries needed
import os			        # For file path operations
import numpy as np		    # For array operations on cell images
import streamlit as st		# For @st.cache_resource decorator
import tensorflow as tf		# For model building and loading

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"	# Disable GPU to ensure compatibility on systems without CUDA & GPU	
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"	# Hide TensorFlow logs (only show errors)
from tensorflow.keras import layers, models	# For model building and loading

# Default model path relative to the project root
# __file__ 							                            = \app\digit_recognizer.py
# os.path.abspath(__file__) 					                = C:\Users\...\project\app\digit_recognizer.py
# os.path.dirname(os.path.abspath(__file__)) 			        = C:\Users\...\project\app
# os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 	= C:\Users\...\project 
# Then join with "model\digit_model.h5"
# Final path: C:\Users\...\project\model\digit_model.h5
_DEFAULT_MODEL_PATH = os.path.join(
	os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model", 
	"digit_model.h5"
)


# ── 1. Model Loading ──────────────────────────────────────────────────────────

@st.cache_resource	    # Decorator from Streamlit that caches the loaded model so it's only loaded once
def load_model(model_path: str = None):
    # Accept model_path as parameter with default None
    # If None, get path from global variable _DEFAULT_MODEL_PATH
    path = model_path or _DEFAULT_MODEL_PATH

    # Handle error of model not found and provide solution to user
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at: {path}\n"
            "Run  python model/train_model.py  to train and save it first."
        )

    # Return the loaded model using TensorFlow's load_model function
    return tf.keras.models.load_model(path)


# ── 2. Grid Predicting ────────────────────────────────────────────────────────

def predict_grid(cells: list, model=None) -> list:
    # Parameters:
    # 1. cells: List of 81 cell images = List of 81 arrays
    #           Each image shape (28,28,1), type float32, values [0,1]    
    # 2. model: tf.keras.models with default None
    #           If None, call load_model()
    if model is None:
        model = load_model()

    # Stack list of 81 arrays into a single array/batch
    # Shape: 81x(28,28,1) -> (81,28,28,1)
    batch = np.stack(cells).astype(np.float32)

    # Convert np.ndarray into tf.constant, shape unchange
    tensor = tf.constant(batch)

    # Pass tensor into the model for prediction but not training
    # Shape: (81,28,28,1) -> (81,10)
    # The model returns probabilities of cell images to be 0-9 digits
    # Convert the returned tf.constant into np.ndarray
    predictions = model(tensor, training=False).numpy()

    # argmax extracts the index of maximum value of each row
    # = predicted class (0–9) of each cell
    # Shape: (81x10) -> (81,)
    digit_classes = np.argmax(predictions, axis=1)

    # Reshape (81,) array into (9,9) grid
    grid = [[int(digit_classes[i * 9 + j]) for j in range(9)] for i in range(9)]

    return grid


# ── 3. Model Building Reference ───────────────────────────────────────────────

def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),	# Input layer: defines the input shape 28x28x1 
        layers.Rescaling(1.0 / 255.0),		# Normalization: scales pixel values from [0,255] to [0,1]

	    # Random image distortion (rotation, shift, zoom)
        layers.RandomRotation(0.05),		    # Randomly rotate image by ±0.05 radians (~±2.8°)
        layers.RandomTranslation(0.10, 0.10),	# Randomly shift image by ±10%
        layers.RandomZoom(0.10),		        # Randomly zoom in/out by ±10%

	    # Conv2D layer: 32 filters of size 3x3
    	# ReLU converts negative values to 0 (introduces non-linearity)
    	# This layer extracts simple features like edges and corners
    	# Output size = Input size - Filter size + 1 = 28-3+1 = 26
	    # padding="same" keeps the output size same as input
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),

	    # MaxPooling: reduces 26x26 to 13x13 (downsampling)
   	    # Keeps the most important features, discards unimportant details
        layers.MaxPooling2D((2, 2)),

	    # Second Conv2D layer: 64 filters of size 3x3
    	# Extracts more complex features like vertices, arcs, and digit parts
    	# Output size = 13-3+1 = 11
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),

    	# MaxPooling: reduces 11x11 to 5x5
        layers.MaxPooling2D((2, 2)),

	    # Flatten: converts 2D feature maps (5x5x64) into 1D vector (5×5×64 = 1600 neurons)
        layers.Flatten(),

	    # Dropout: randomly drops 50% of neurons during training to prevent overfitting
        layers.Dropout(0.5),

	    # Dense (fully connected) layer: 128 neurons
        layers.Dense(128, activation="relu"),

	    # Output layer: 10 neurons (digits 0-9)
	    # Softmax outputs probability distribution
        layers.Dense(10, activation="softmax"),
    ], name="digit_cnn")

    model.compile(
        optimizer="adam",			        # Optimizer: adaptive learning rate
        loss="categorical_crossentropy",	# Loss function: suitable for one-hot labels
        metrics=["accuracy"],			    # Metric: track accuracy during training
    )
    return model
