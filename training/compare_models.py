import pickle
import time
import tensorflow as tf  # type: ignore
from tensorflow.keras.preprocessing.image import (  # type: ignore
    ImageDataGenerator)
from tabulate import tabulate   # type: ignore


# Load training histories
def load_training_history(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


# Paths to models and histories
models = {
    "CNN": {
        "history_path": "training_history_improved.pkl",
        "model_path": "asl_fingerspell_cnn_improved.keras"
    },
    "MobileNetV2 (Frozen)": {
        "history_path": "training_history_mobilenet.pkl",
        "model_path": "asl_fingerspell_mobilenet.keras"
    },
    "MobileNetV2 (Fine-Tuned)": {
        "history_path": "training_history_mobilenet_finetuned.pkl",
        "model_path": "asl_fingerspell_mobilenet_finetuned.keras"
    },
    "ResNet50": {
        "history_path": "training_history_resnet.pkl",
        "model_path": "asl_fingerspell_resnet.keras"
    },
    "EfficientNetB0": {
        "history_path": "training_history_efficientnet.pkl",
        "model_path": "asl_fingerspell_efficientnet.keras"
    },
}

# Prepare the test data generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    "../dataset/Test_Alphabet",
    target_size=(128, 128),
    batch_size=32,
    class_mode="sparse",
    shuffle=False  # Ensures consistency in evaluation
)

# Store results
results = []

for model_name, paths in models.items():
    print(f"Evaluating {model_name}...")

    # Load training history
    history = load_training_history(paths["history_path"])

    # Get final validation accuracy
    final_val_acc = history["val_accuracy"][-1]

    # Load model & measure test accuracy
    model = tf.keras.models.load_model(paths["model_path"])

    # Measure evaluation time
    start_time = time.time()
    test_loss, test_acc = model.evaluate(test_generator)
    eval_time = time.time() - start_time

    # Store results
    results.append([model_name, final_val_acc * 100, test_acc * 100,
                    eval_time])

# Print comparison table
headers = ["Model", "Validation Accuracy (%)", "Test Accuracy (%)",
           "Evaluation Time (s)"]
print("\nModel Comparison:\n")
print(tabulate(results, headers=headers, tablefmt="grid"))
