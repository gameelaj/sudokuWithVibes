# =============================================================================
# train_model.py — CNN Training Script for Digit Recognition
# =============================================================================
# Member 5 runs this to train the digit recognition CNN and save the model.
#
# Prerequisites:
#   - Run prepare_cells.py first to generate data/cells/
#
# What this script does:
#   1. Loads cell images from data/cells/ using image_dataset_from_directory
#   2. Applies data augmentation to improve real-photo performance
#   3. Builds a CNN: Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dense
#   4. Trains with class weights to handle class-0 imbalance
#   5. Saves the trained model to model/digit_model.h5
#   6. Saves accuracy/loss plots to model/training_plots.png
#
# Run from the project root:
#   python model/train_model.py
#
# Tested on: TensorFlow 2.21
# Dependencies: tensorflow>=2.10, numpy, matplotlib, scikit-learn
# Source (image_dataset_from_directory):
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
# Source (CNN baseline):
# https://www.tensorflow.org/tutorials/quickstart/advanced
# Source (class_weight imbalance):
# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
# =============================================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers with no display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

# ── Configuration ─────────────────────────────────────────────────────────────
CELLS_DIR   = "data/cells"
MODEL_DIR   = "model"
MODEL_PATH  = os.path.join(MODEL_DIR, "digit_model.h5")
PLOTS_PATH  = os.path.join(MODEL_DIR, "training_plots.png")

IMG_SIZE    = (28, 28)
BATCH_SIZE  = 64
EPOCHS      = 20
NUM_CLASSES = 10    # 0 (empty) + digits 1–9
SEED        = 42

os.makedirs(MODEL_DIR, exist_ok=True)


# ── 1. Data Loading ───────────────────────────────────────────────────────────

def build_datasets():
    """
    Load train and val cell images using the modern TF 2.x API.

    image_dataset_from_directory replaces the deprecated ImageDataGenerator.
    It reads class labels from subfolder names (0/ through 9/) automatically.

    Returns raw datasets (not yet augmented or normalised).
    Normalisation and augmentation are applied as model layers instead,
    which is the recommended TF 2.x approach.

    # Source: https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(CELLS_DIR, "train"),
        labels="inferred",          # class = subfolder name (0–9)
        label_mode="categorical",   # one-hot vectors for softmax output
        color_mode="grayscale",     # single channel → shape (28, 28, 1)
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(CELLS_DIR, "val"),
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Print class mapping so we can verify order matches 0–9
    print(f"  Class names (folder order): {train_ds.class_names}")

    # Performance optimisation: cache in memory + prefetch next batch while GPU trains
    # Source: https://www.tensorflow.org/guide/data_performance
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


# ── 2. Class Weights ──────────────────────────────────────────────────────────

def compute_weights(train_ds) -> dict:
    """
    Compute per-class weights to counteract the class-0 dominance.
    Class 0 (empty) makes up ~47% of cells — without weighting, the model
    would be biased toward predicting empty for ambiguous cells.

    We collect all labels from the dataset first, then use sklearn's
    compute_class_weight to calculate balanced weights.

    Returns a dict {class_index: weight} for model.fit().

    # Source: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    """
    all_labels = []
    for _, label_batch in train_ds.unbatch():
        all_labels.append(int(tf.argmax(label_batch).numpy()))

    all_labels     = np.array(all_labels)
    unique_classes = np.unique(all_labels)
    weights        = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=all_labels,
    )
    weight_dict = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}

    print("\n  Class weights:")
    for cls, w in sorted(weight_dict.items()):
        label = "empty" if cls == 0 else f"digit {cls}"
        print(f"    Class {cls} ({label:<8}): {w:.4f}")

    return weight_dict


# ── 3. Model Architecture ─────────────────────────────────────────────────────

def build_model() -> tf.keras.Model:
    """
    CNN with data augmentation baked in as the first layers.

    Putting augmentation inside the model means:
      - It only runs during training (automatically disabled at inference time)
      - No changes needed in digit_recognizer.py — Member 3 just calls model.predict()

    Input:  (28, 28, 1)  — grayscale cell image, raw uint8 0–255
    Output: (10,)        — softmax probabilities for classes 0–9

    Architecture:
      Rescaling   → normalise [0,255] to [0,1]
      Augmentation→ random rotation / zoom / shift (training only)
      Conv2D(32)  → detect edges and basic strokes
      MaxPool     → halve spatial dimensions to (14,14,32)
      Conv2D(64)  → detect digit-level features
      MaxPool     → halve again to (7,7,64)
      Flatten     → 3136-dim vector
      Dropout(0.5)→ regularisation
      Dense(128)  → learned feature combination
      Dense(10)   → softmax class probabilities

    # Source: https://www.tensorflow.org/tutorials/quickstart/advanced
    # Source: https://keras.io/api/layers/preprocessing_layers/
    """
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Normalisation — converts uint8 [0,255] → float32 [0,1]
        # Source: https://keras.io/api/layers/preprocessing_layers/image_preprocessing/rescaling/
        layers.Rescaling(1.0 / 255.0),

        # Data augmentation — only active during model.fit(), not predict()
        # Source: https://www.tensorflow.org/tutorials/images/data_augmentation
        layers.RandomRotation(0.05),           # ±~10 degrees
        layers.RandomTranslation(0.10, 0.10),  # ±10% shift
        layers.RandomZoom(0.10),               # ±10% zoom

        # Block 1: low-level features (edges, curves)
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Block 2: mid-level features (digit strokes)
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Classifier head
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="digit_cnn")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


# ── 4. Training ───────────────────────────────────────────────────────────────

def train(model, train_ds, val_ds, class_weights):
    """
    Train the model with early stopping and learning rate reduction.

    Callbacks:
      EarlyStopping     — stops if val_accuracy doesn't improve for 5 epochs,
                          restores the best weights automatically.
      ReduceLROnPlateau — halves learning rate if val_loss stalls for 3 epochs.

    # Source: https://keras.io/api/callbacks/early_stopping/
    # Source: https://keras.io/api/callbacks/reduce_lr_on_plateau/
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ── 5. Evaluation ─────────────────────────────────────────────────────────────

def evaluate(model, val_ds):
    """
    Print final validation accuracy and loss.
    Target per briefing: > 98% accuracy on the validation set.
    """
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\n  Final validation accuracy: {acc * 100:.2f}%")
    print(f"  Final validation loss:     {loss:.4f}")
    if acc >= 0.98:
        print("  ✅ Target accuracy (>98%) achieved.")
    else:
        print("  ⚠️  Below 98% — consider more epochs or stronger augmentation.")
    return acc


# ── 6. Save Plots ─────────────────────────────────────────────────────────────

def save_plots(history, save_path: str):
    """
    Save training/validation accuracy and loss curves as a PNG.
    Member 6 will include this in the presentation slides.

    # Source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/index.html
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"],     label="Train", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val",   linewidth=2, linestyle="--")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["loss"],     label="Train", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Val",   linewidth=2, linestyle="--")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training plots saved to {save_path}")


# ── 7. Save Model ─────────────────────────────────────────────────────────────

def save_model(model, path: str):
    """
    Save the trained model in HDF5 format.
    Member 3 loads this in digit_recognizer.py.

    Contract (must match contracts.py):
      Input shape:  (None, 28, 28, 1)
      Output shape: (None, 10)
      File:         model/digit_model.h5
    """
    model.save(path)
    print(f"  Model saved to {path}")
    print(f"  Input shape:  {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PhotoSudoku — CNN Training Pipeline (TF 2.21)")
    print("=" * 60)

    train_dir = os.path.join(CELLS_DIR, "train")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Preprocessed cells not found at {train_dir}.\n"
            "Run  python model/prepare_cells.py  first."
        )

    print("\n[1/6] Loading datasets...")
    train_ds, val_ds = build_datasets()

    print("\n[2/6] Computing class weights...")
    class_weights = compute_weights(train_ds)

    print("\n[3/6] Building model...")
    model = build_model()

    print("\n[4/6] Training...")
    history = train(model, train_ds, val_ds, class_weights)

    print("\n[5/6] Evaluating...")
    evaluate(model, val_ds)

    print("\n[6/6] Saving outputs...")
    save_plots(history, PLOTS_PATH)
    save_model(model, MODEL_PATH)

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Model → {MODEL_PATH}")
    print(f"  Plots → {PLOTS_PATH}")
    print("  Member 3 can now use digit_model.h5 in digit_recognizer.py")
    print("=" * 60)


if __name__ == "__main__":
    main()