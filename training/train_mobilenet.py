import sys  # type: ignore
import os
import pickle
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.applications import MobileNetV2   # type: ignore
from tensorflow.keras.layers import (   # type: ignore
    GlobalAveragePooling2D, Dense, Dropout)
from tensorflow.keras.optimizers import Adam  # type: ignore

# Get the parent directory and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.dataset_loader import create_generators  # noqa: E402

# Load dataset
train_dataset, val_dataset, test_dataset = create_generators()


# Define MobileNetV2 Model
def create_mobilenet_model() -> Sequential:
    base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False,
                             weights="imagenet")
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(27, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Define Fine-Tuned MobileNetV2 Model
def create_finetuned_mobilenet() -> Sequential:
    base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False,
                             weights="imagenet")

    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(27, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


model = create_finetuned_mobilenet()
model.summary()

EPOCHS = 300
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# Save Model
model.save("Models/asl_fingerspell_mobilenet_finetuned_new_dataset.keras")

# Save Training History
history_path = "Training History/training_history_mobilenet_finetuned_new_dataset.pkl"
with open(history_path, "wb") as f:
    pickle.dump(history.history, f)

print(f"Training history saved to {history_path}.")

test_loss, test_acc = model.evaluate(test_dataset)
print(f"\nFine-Tuned MobileNetV2 Test Accuracy: {test_acc:.4f}")
