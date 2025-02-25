import sys
import os
import pickle
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import (   # type: ignore
    Dense, Dropout, GlobalAveragePooling2D)
from tensorflow.keras.applications import ResNet50  # type: ignore
from tensorflow.keras.optimizers import Adam    # type: ignore

# Get the parent directory and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.dataset_loader import create_generators  # noqa: E402

# Load dataset
train_generator, val_generator, test_generator = create_generators()


# Define ResNet50 Model
def create_resnet_model() -> Sequential:
    """
    Creates a fine-tuned ResNet50 model.

    Returns:
        A compiled Keras Sequential model.
    """
    base_model = ResNet50(input_shape=(128, 128, 3), include_top=False,
                          weights="imagenet")

    # Unfreeze the last 30 layers for fine-tuning
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    # Add classification layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(27, activation="softmax")
    ])

    # Compile model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Create and summarize the model
model = create_resnet_model()
model.summary()

# Train the Model
EPOCHS = 15  # Fine-tuning usually requires fewer epochs
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Save ResNet Model
model.save("asl_fingerspell_resnet.keras")

# Save Training History
history_path = "training_history_resnet.pkl"
with open(history_path, "wb") as f:
    pickle.dump(history.history, f)

print(f"Training history saved to {history_path}.")

# Evaluate on Test Set
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nResNet50 Test Accuracy: {test_acc:.4f}")
