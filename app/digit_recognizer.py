# ============================================
# Module 3: Digit Recognizer
# Owner: Member 3

# This module loads the trained CNN model and uses it to recognize digits
# from 81 cell images extracted from a Sudoku photo.
# ============================================

# ============================================
# 1. Import Libraries
# ============================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model as keras_load_model

# ============================================
# 2. load_model()
# ============================================

# Source: https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model
def load_model(model_path: str = "model/digit_model.h5"):
    """Load trained CNN model from disk. Call once at app startup."""
    
    model = keras_load_model(model_path)
    return model
    
# ============================================
# 3. build_model()
# ============================================

# Source: https://www.tensorflow.org/tutorials/quickstart/advanced
def build_model() -> tf.keras.Model:
    """Build and compile CNN model for digit recognition (28x28x1 input, 10-class output)."""
    
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ], name="digit_cnn")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
    
# ============================================
# 4. predict_grid()
# ============================================

# Source: https://numpy.org/doc/stable/reference/generated/numpy.stack.html
# Source: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
def predict_grid(cells: list[np.ndarray], model) -> list[list[int]]:
    """Predict 81 cell images, return 9x9 digit grid. 0 = empty cell."""
    
    # Step 1: Stack all 81 cells into a single batch
    # Input: list of 81 arrays, each shape (28, 28, 1)
    # Output: one array of shape (81, 28, 28, 1)
    batch = np.stack(cells)
    
    # Step 2: Run model prediction on the entire batch
    # Output shape: (81, 10) — 10 classes (0-9) for each cell
    predictions = model.predict(batch)
    
    # Step 3: Get the predicted class for each cell (0-9)
    # np.argmax returns the index of the highest probability
    # Output shape: (81,) — one integer per cell
    digit_classes = np.argmax(predictions, axis=1)
    
    # Step 4: Reshape the flat list of 81 digits into a 9x9 grid
    # List comprehension: for each row i (0-8), for each column j (0-8)
    # digit_classes[i*9 + j] gives the j-th element of row i
    grid = [[int(digit_classes[i*9 + j]) for j in range(9)] for i in range(9)]
    
    return grid
