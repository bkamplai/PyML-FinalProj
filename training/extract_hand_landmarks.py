import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=6,  # Increased from 4
    min_detection_confidence=0.05,  # Reduced from 0.1
    min_tracking_confidence=0.05
)

# Define dataset paths
DATASET_PATH = "../dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "Train_Alphabet")
TEST_PATH = os.path.join(DATASET_PATH, "Test_Alphabet")
SKIPPED_DIR = "skipped_images"
LOW_CONF_DIR = "low_confidence_images"
os.makedirs(SKIPPED_DIR, exist_ok=True)
os.makedirs(LOW_CONF_DIR, exist_ok=True)

# Store confidence scores for skipped images
skipped_confidences = []
low_confidence_images = []


def adaptive_gamma_correction(image, gamma=1.0):
    """Dynamically adjust gamma correction based on brightness levels."""
    mean_intensity = np.mean(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    gamma = 1.5 if mean_intensity < 100 else (0.7 if mean_intensity > 180
                                              else 1.0)
    table = np.array([(i / 255.0) ** gamma * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Apply Adaptive Gamma Correction
    image = adaptive_gamma_correction(image)

    # Convert back to RGB for MediaPipe processing
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return image


def extract_landmarks(image_path: str, class_name: str):
    """Extract 21 hand landmarks and confidence estimation from an image."""
    if class_name.lower() == "blank":
        return None  # Skip 'Blank' class images immediately

    image = preprocess_image(image_path)
    if image is None:
        return None

    results = hands.process(image)

    if not results.multi_hand_landmarks:
        cv2.imwrite(os.path.join(SKIPPED_DIR, os.path.basename(image_path)),
                    image)
        return None

    landmarks = []
    # Take the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])

    # Compute estimated confidence
    # Adjusted for 6 hands max
    avg_confidence = len(results.multi_hand_landmarks) / 6

    # Handle Low-Confidence Cases
    if avg_confidence < 0.2:
        low_confidence_images.append([class_name, image_path, avg_confidence])
        cv2.imwrite(os.path.join(LOW_CONF_DIR, os.path.basename(image_path)),
                    image)

    return [class_name, image_path, avg_confidence] + landmarks


def process_dataset(dataset_path: str, dataset_type: str):
    """Process images in dataset to extract hand landmarks."""
    data = []
    skipped_count = 0

    for class_name in tqdm(sorted(os.listdir(dataset_path)),
                           desc=f"Processing {dataset_type}"):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        if class_name.lower() == "blank":
            print(f"Skipping 'Blank' class: {class_dir}")
            continue  # Skip "Blank" images

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            result = extract_landmarks(img_path, class_name)

            if result:
                data.append(result)
            else:
                skipped_count += 1

    print(f"⚠️ Skipped {skipped_count} images (excluding 'Blank' class).")
    return data


# Process dataset
train_data = process_dataset(TRAIN_PATH, "Train")
test_data = process_dataset(TEST_PATH, "Test")

# Define DataFrame columns
columns = ["Class", "Filename", "Confidence"] + \
    [f"LM_{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
train_df = pd.DataFrame(train_data, columns=columns)
test_df = pd.DataFrame(test_data, columns=columns)

# Save extracted landmarks
train_df.to_csv("train_hand_landmarks.csv", index=False)
test_df.to_csv("test_hand_landmarks.csv", index=False)

# Save Low-Confidence Images for Review
low_conf_df = pd.DataFrame(low_confidence_images,
                           columns=["Class", "Filename", "Confidence"])
low_conf_df.to_csv("low_confidence_images.csv", index=False)

# Final logging
print("\n✅ Hand landmark extraction complete!")
print(f"📂 Train dataset size: {len(train_df)} images")
print(f"📂 Test dataset size: {len(test_df)} images")
print(f"⚠️ Skipped images saved in '{SKIPPED_DIR}'")
print(f"🟡 Low-confidence images saved in '{LOW_CONF_DIR}'")

if skipped_confidences:
    print(f"🔍 Average confidence of skipped images: {np.mean(
        skipped_confidences):.4f}")
