import sys  # type: ignore
import os
import pickle
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D    # type: ignore
from tensorflow.keras.layers import Flatten, Dense, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

# Get the parent directory and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.dataset_loader import create_generators

# Load data generators
train_generator, val_generator, test_generator = create_generators()


# Define CNN Model
def create_cnn_model() -> Sequential:
    """
    Creates a baseline CNN model for ASL fingerspelling classification.

    Returns:
        A compiled Keras Sequential model.
    """
    model = Sequential([
        # Convolutional Layer 1
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),

        # Convolutional Layer 2
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        # Convolutional Layer 3
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        # Flatten the feaure maps
        Flatten(),

        # Fully Connected Layer
        Dense(128, activation="relu"),
        Dropout(0.5),   # Regularization to prevent overfitting
        Dense(27, activation="softmax")  # 27 classes (A-z + Blank)
    ])

    # Compile Model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Create and summarize the model
model = create_cnn_model()
model.summary()

# Train the Model
EPOCHS = 10  # Start with 10, can increase later
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Save Model
model.save("asl_fingerspell_cnn.keras")

# Evaluae on Test Set
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Save training history
history_path = "training_history.pkl"
with open(history_path, "wb") as f:
    pickle.dump(history.history, f)

print(f"Training history saved to {history_path}.")
