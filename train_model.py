# ============================================
# Environment: python=3.10, tensorflow=2.21.0
# ============================================

# ============================================
# 1. Import Libraries
# ============================================

# 1.1 Import TensorFlow library
import tensorflow as tf

# Print TensorFlow version to confirm successful installation
print(tf.__version__)

# 1.2 Import additional modules for data loading, model building, and utilities
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================
# 2. Load and Preprocess Data
# ============================================

# 2.1 Download MNIST dataset (Modified National Institute of Standards and Technology)

# x = input images (handwritten digits), y = correct labels (0-9)
# Training set: 60,000 images, Test set: 10,000 images
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

print("Shape of dataset:")
print(f"x_train = {x_train.shape}, y_train = {y_train.shape}")
print(f"x_test = {x_test.shape}, y_test = {y_test.shape}")
# (number_of_data, size_of_picture: 28x28 pixels)

# 2.2 Display first 5 training images to verify data

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)          # Create 1 row × 5 columns grid, select i-th position
    plt.imshow(x_train[i], cmap='gray')  # Show image in grayscale (not color map)
    plt.title(f"Label: {y_train[i]}")    # Show correct answer as title
    plt.axis('off')                 # Hide axes for cleaner look
plt.show()

# 2.3 Replace digit 0 by black images

# For Sudoku, we don't need digit 0 (empty cells are class 0, not digit zero)
# Create a boolean mask: True for digits 1-9, False for digit 0
# Example: mask_train = [True, False, True, True, True, ...]
#              labels = [5,    0,     4,    1,    9,    ...]
mask_train = y_train != 0
mask_test = y_test != 0

# Keep only samples where mask is True (remove all digit 0 images)
x_train = x_train[mask_train]
y_train = y_train[mask_train]
x_test = x_test[mask_test]
y_test = y_test[mask_test]

print(f"After removing digit 0 - Train: {x_train.shape}, Test: {x_test.shape}")

# Sudoku has empty cells, which need to be recognized as class 0
# Create black images to simulate empty cells
num_blank_train = 60000-len(x_train)
num_blank_test = 10000-len(x_test)

# Generate black images
blank_train = np.zeros((num_blank_train, 28, 28))
blank_test = np.zeros((num_blank_test, 28, 28))

# All blank cells have label 0
blank_y_train = np.zeros(num_blank_train, dtype=int)
blank_y_test = np.zeros(num_blank_test, dtype=int)

# Merge blanks to the dataset
x_train = np.concatenate([x_train, blank_train])
y_train = np.concatenate([y_train, blank_y_train])
x_test = np.concatenate([x_test, blank_test])
y_test = np.concatenate([y_test, blank_y_test])

print(f"Combined - Train: {x_train.shape}, Test: {x_test.shape}")

# Shuffle the dataset after combine
indices = np.random.permutation(len(x_train))
x_train = x_train[indices]
y_train = y_train[indices]

# 2.4 Reshape dataset to meet Conv2D (2-Dimensional Convolutional Neural Network) requirements

# Normalization: scale pixel values from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Conv2D expects input shape: (batch, height, width, channels)
# channels = 1 for grayscale, 3 for RGB color images
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 2.5 Dataset after preprocessing

print(f"After preprocessing - Train: {x_train.shape}, Test: {x_test.shape}")

print(f"Training set class distribution:")
print(f"Class 0 (empty): {np.sum(y_train == 0)}")
for digit in range(1, 10):
    print(f"\tClass {digit}: {np.sum(y_train == digit)}")

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)          # Create 1 row × 5 columns grid, select i-th position
    plt.imshow(x_train[i], cmap='gray')  # Show image in grayscale (not color map)
    plt.title(f"Label: {y_train[i]}")    # Show correct answer as title
    plt.axis('off')                 # Hide axes for cleaner look
plt.show()

# ============================================
# 3. Build CNN Model
# ============================================

model = models.Sequential([
    # Input layer: defines the input shape 28x28x1 (grayscale)
    layers.Input(shape=(28, 28, 1)),
    
    # Conv2D layer: 32 filters of size 3x3
    # ReLU converts negative values to 0 (introduces non-linearity)
    # This layer extracts simple features like edges and corners
    # Output size = Input size - Filter size + 1 = 28-3+1 = 26
    layers.Conv2D(32, (3, 3), activation='relu'),
    
    # MaxPooling: reduces 26x26 to 13x13 (downsampling)
    # Keeps the most important features, discards unimportant details
    layers.MaxPooling2D((2, 2)), 
    
    # Second Conv2D layer: 64 filters of size 3x3
    # Extracts more complex features like vertices, arcs, and digit parts
    # Output size = 13-3+1 = 11
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # MaxPooling: reduces 11x11 to 5x5
    layers.MaxPooling2D((2, 2)), 
    
    # Flatten: converts 2D feature maps (5x5x64) into 1D vector (5×5×64 = 1600 neurons)
    layers.Flatten(),
    
    # Dense (fully connected) layer: 64 neurons
    layers.Dense(64, activation='relu'),
    
    # Output layer: 10 neurons (digits 0-9), softmax outputs probability distribution
    layers.Dense(10, activation='softmax')
])

# Show structure of model
model.summary()

# ============================================
# 4. Compile Model
# ============================================

model.compile(
    optimizer='adam',                       # Optimizer: adaptive learning rate
    loss='sparse_categorical_crossentropy', # Loss function: suitable for integer labels
    metrics=['accuracy']                    # Metric: track accuracy during training
)

# ============================================
# 5. Train Model
# ============================================

history = model.fit(
    # Use model to run on Training set
    x_train, y_train,
    
    # Train for 5 epochs (rounds)
    # Each epoch processes 60,000/32 = 1,875 batches
    # Process 32 images per batch (parallel processing)
    epochs=5,                
    batch_size=32, 

    # Reserve 10% (6,000 images) for validation
    # Validation happens at the end of each epoch (not during training)
    validation_split=0.1     
)

# ============================================
# 6. Evaluate Model
# ============================================

# Use model to run on Test set 
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Accuracy of Test: {test_acc:.4f} ({test_acc*100:.2f}%)")

# ============================================
# 7. Save Model
# ============================================

model.save("model/digit_model.h5")

# Verify whether the model have been successfully saved
if os.path.exists("model/digit_model.h5"):
    file_size = os.path.getsize("model/digit_model.h5") / (1024*1024)
    print(f"Model file size: {file_size:.2f} MB")
else:
    print("Save unsuccessful.")

# ============================================
# 8. Plot Training History
# ============================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()