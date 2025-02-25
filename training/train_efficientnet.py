import sys
import os
import pickle
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import (   # type: ignore
    Dense, Dropout, GlobalAveragePooling2D)
from tensorflow.keras.applications import EfficientNetB0    # type: ignore
from tensorflow.keras.optimizers import Adam    # type: ignore

# Get the parent directory and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.dataset_loader import create_generators  # noqa: E402

# Load dataset
train_generator, val_generator, test_generator = create_generators()


# Define EfficientNetB0 Model
def create_efficientnet_model() -> Sequential:
    """
    Creates a fine-tuned EfficientNetB0 model.

    Returns:
        A compiled Keras Sequential model.
    """
    base_model = EfficientNetB0(input_shape=(128, 128, 3), include_top=False,
                                weights="imagenet")

    # Unfreeze the last 20 layers for fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    # Add custom classification layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(27, activation="softmax")
    ])

    # Compile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Create and summarize the model
model = create_efficientnet_model()
model.summary()

# Train the Model
EPOCHS = 15  # EfficientNet usually needs fewer epochs
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Save EfficientNet Model
model.save("asl_fingerspell_efficientnet.keras")

# Save Training History
history_path = "training_history_efficientnet.pkl"
with open(history_path, "wb") as f:
    pickle.dump(history.history, f)

print(f"Training history saved to {history_path}.")

# Evaluate on Test Set
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nEfficientNetB0 Test Accuracy: {test_acc:.4f}")
