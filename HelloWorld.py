import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load and Preprocess Data ---
# For Handwritten Digit Recognition, we'll use the MNIST dataset.
# EMNIST would be used for characters, but it's larger and requires more complex handling.
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Reshape the images to add a channel dimension (for grayscale images, it's 1)
# CNNs expect input in the format (batch_size, height, width, channels)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Convert labels to one-hot encoding
# e.g., if y_train is 3, it becomes [0,0,0,1,0,0,0,0,0,0] for 10 classes (digits 0-9)
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# --- 2. Build the CNN Model ---
print("\nBuilding CNN model...")
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)), # Input layer expects 28x28 grayscale images
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"), # 32 filters, 3x3 kernel
        layers.MaxPooling2D(pool_size=(2, 2)), # Reduce spatial dimensions
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"), # 64 filters, 3x3 kernel
        layers.MaxPooling2D(pool_size=(2, 2)), # Reduce spatial dimensions
        layers.Flatten(), # Flatten the 2D feature maps into a 1D vector
        layers.Dropout(0.5), # Dropout for regularization to prevent overfitting
        layers.Dense(num_classes, activation="softmax"), # Output layer with softmax for classification
    ]
)

model.summary()

# --- 3. Compile the Model ---
print("\nCompiling model...")
model.compile(
    loss="categorical_crossentropy", # Suitable for multi-class classification with one-hot labels
    optimizer="adam", # Adam optimizer is a good general-purpose choice
    metrics=["accuracy"], # Monitor accuracy during training
)

# --- 4. Train the Model ---
print("\nTraining model...")
batch_size = 128
epochs = 10 # You can increase this for better accuracy, but it takes longer

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1, # Use 10% of training data for validation
    verbose=1 # Show progress during training
)

# --- 5. Evaluate the Model ---
print("\nEvaluating model...")
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")

# --- 6. Visualize Training History (Optional) ---
print("\nPlotting training history...")
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()