import time
import pickle
import tensorflow as tf  # type: ignore
import numpy as np
import cv2

# Load label mapping
with open('../preprocessing/label_mapping.pkl', 'rb') as f:
    label_to_index = pickle.load(f)
index_to_label = {v: k for k, v in label_to_index.items()}

# Load test image (replace with a real ASL image)
TEST_IMAGE_PATH = 'test_sample.png'  # Provide a real test image
IMAGE_SIZE = (128, 128)


def preprocess_image(image_path: str) -> np.ndarray:
    """Loads and preprocesses an image for model inference."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0   # Normalize
    return np.expand_dims(img, axis=0)  # Add batch dimension


def load_model(model_path: str) -> tf.keras.Model:
    """Loads a trained Keras model."""
    return tf.keras.models.load_model(model_path)


def test_model_inference(model_path: str, test_image: np.ndarray,
                         model_name: str):
    """Tests inference speed of a given model."""
    model = load_model(model_path)

    # Measure inference time
    start_time = time.time()
    predictions = model.predict(test_image)
    end_time = time.time()

    # Get predicted label
    predicted_class = np.argmax(predictions)
    predicted_label = index_to_label[predicted_class]
    inference_time = end_time - start_time

    print(f"Model: {model_name} | Prediction: {predicted_label} | " +
          f"Time: {inference_time:.4f}s")
    return inference_time


# Load test image
test_image = preprocess_image(TEST_IMAGE_PATH)

# Run inference tests
mobile_v2_time = test_model_inference("Models/asl_fingerspell_mobilenet_finetuned.keras",
                                      test_image, "Fine-Tuned MobileNetV2")
resnet_time = test_model_inference("Models/asl_fingerspell_resnet.keras", test_image,
                                   "ResNet50")

# Print final comparison
print("\nFINAL INFERENCE SPEED RESULTS:")
print(f"Fine-Tuned MobileNetV2: {mobile_v2_time:.4f}s")
print(f"ResNet50: {resnet_time:.4f}s")

# Determine final recommendation
if mobile_v2_time < resnet_time:
    print("\nFinal Decision: Fine-Tuned MobileNetV2 (Faster for Real-Time" +
          " Inference)")
else:
    print("\nFinal Decision: ResNet50 (Better Accuracy, but Slightly Slower)")
