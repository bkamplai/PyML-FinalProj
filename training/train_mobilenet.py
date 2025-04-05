import sys  # type: ignore
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Get the parent directory and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.dataset_loader import create_generators

# Load dataset
train_generator, val_generator, test_generator = create_generators()


# Define MobileNetV2 Model
def create_mobilenet_model() -> Sequential:
    """
    Creates a Transfer Learning model using MobileNetV2.

    Returns:
        A compiled Keras Sequential model.
    """
    # Load MobileNetV2 as the base model (pre-trained on ImageNet)
    base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False,
                             weights="imagenet")

    # Freeze the base model layers (so we only train our custom layers)
    base_model.trainable = False

    # Add custom classification layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),   # Reduce feature maps to a vector
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(27, activation="softmax")  # 27 classes (A-Z + Blank)
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0005),   # Slower learning rate
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Define Fine-Tuned MobileNetV2 Model
def create_finetuned_mobilenet() -> Sequential:
    """
    Creates a fine-tuned MobileNetV2 model by unfreezing some layers.

    Returns:
        A compiled Keras Sequential model.
    """
    # Load MobileNetV2 as the base model
    base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False,
                             weights="imagenet")

    # Unfreeze the last few layers for fine-tuning
    for layer in base_model.layers[-20:]:  # Unfreezing the last 20 layers
        layer.trainable = True

    # Add custom classification layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),  # Reduce feature maps to a vector
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(27, activation="softmax")  # 27 classes (A-Z + Blank)
    ])

    # Compile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Lower LR for gradual updates
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Create and summarize the model
model = create_finetuned_mobilenet()
model.summary()

# Train the Model
EPOCHS = 10  # Transfer learning converges faster, so fewer epochs
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Save Model
model.save("Models/asl_fingerspell_mobilenet_finetuned_new_dataset.keras")

# Save Training History
history_path = "Training History/training_history_mobilenet_finetuned_new_dataset.pkl"
with open(history_path, "wb") as f:
    pickle.dump(history.history, f)

print(f"Training history saved to {history_path}.")

# Evaluate on Test Set
test_loss, test_acc = model.evaluate(test_generator)
print(f"\n Fine-Tuned MobileNetV2 Test Accuracy: {test_acc:.4f}")
