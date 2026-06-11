# =============================================================================
# app/digit_recognizer.py — CNN Digit Recognition
# Owner: Member 3
#
# Loads the trained model (model/digit_model.h5) once at startup and runs
# batch inference on the 81 cell images produced by image_processor.py.
#
# predict_grid() returns a 9×9 list of ints (0 = empty, 1-9 = digit).
# =============================================================================

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import streamlit as st

# Default model path relative to the project root
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model", "digit_model.h5"
)


# ── Model Loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_path: str = None):
    """
    Load the trained CNN from disk. Results are cached globally so subsequent
    calls return the same object without re-loading from disk.

    Parameters
    ----------
    model_path : str, optional
        Absolute or relative path to the .h5 model file.
        Defaults to model/digit_model.h5 relative to the project root.

    Returns
    -------
    tf.keras.Model
    """
    import tensorflow as tf
    path = model_path or _DEFAULT_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at: {path}\n"
            "Run  python model/train_model.py  to train and save it first."
        )
    return tf.keras.models.load_model(path)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_grid(cells: list, model=None) -> list:
    """
    Predict the digit in each of the 81 cell images.

    Parameters
    ----------
    cells : list[np.ndarray]
        Flat list of 81 arrays, each shape (28, 28, 1), float32, values [0, 1].
        Produced by image_processor.extract_cells().
    model : tf.keras.Model, optional
        Pre-loaded model. If None, load_model() is called automatically.

    Returns
    -------
    list[list[int]]
        9×9 grid of integers. 0 = empty cell, 1-9 = recognised digit.
    """
    if model is None:
        model = load_model()

    import tensorflow as tf

    # Stack into a single batch: (81, 28, 28, 1)
    batch = np.stack(cells).astype(np.float32)

    tensor = tf.constant(batch)

    # Use direct call instead of model.predict() to avoid Streamlit/Metal deadlocks.
    predictions = model(tensor, training=False).numpy()

    # argmax gives the predicted class (0–9) per cell
    digit_classes = np.argmax(predictions, axis=1)

    # Reshape flat 81-element array into 9×9 grid
    grid = [[int(digit_classes[i * 9 + j]) for j in range(9)] for i in range(9)]
    return grid


# ── Standalone build helper (for training reference) ─────────────────────────

def build_model():
    """
    Build and compile a fresh CNN with the same architecture as the saved model.
    Useful for reference or re-training without running train_model.py directly.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Rescaling(1.0 / 255.0),
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomZoom(0.10),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ], name="digit_cnn")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
