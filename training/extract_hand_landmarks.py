import os
import cv2
import mediapipe as mp  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from tqdm import tqdm   # type: ignore

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2,
    min_detection_confidence=0.1,  # Adjusted threshold
    min_tracking_confidence=0.1
)

# Define dataset paths
DATASET_PATH = "../dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "Train_Alphabet")
TEST_PATH = os.path.join(DATASET_PATH, "Test_Alphabet")
SKIPPED_DIR = "skipped_images"
LOW_CONFIDENCE_DIR = "low_confidence_images"
os.makedirs(SKIPPED_DIR, exist_ok=True)
os.makedirs(LOW_CONFIDENCE_DIR, exist_ok=True)

# Store confidence scores
skipped_confidences: list = []
low_confidence_images = []


# Preprocessing Functions
def adjust_gamma(image, gamma=1.0):
    """Apply gamma correction to adjust brightness."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_clahe(image):
    """Apply CLAHE for contrast enhancement."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def sharpen_image(image):
    """Apply sharpening filter to enhance details."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def preprocess_image(image_path):
    """
    Preprocess image with adaptive brightness correction, contrast
    enhancement, and sharpening.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Compute brightness and contrast
    brightness = np.mean(image)
    contrast = image.std()

    # Apply gamma correction for dark images
    if brightness < 50:
        image = adjust_gamma(image, gamma=1.5)  # Brighten dark images
    elif brightness > 180:
        # Reduce brightness if too bright
        image = adjust_gamma(image, gamma=0.7)

    # Apply CLAHE for low-contrast images
    if contrast < 20:
        image = apply_clahe(image)

    # Sharpen image
    image = sharpen_image(image)

    return image


# Hand Landmark Extraction
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
    # Process only the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])

    # Estimate confidence based on detection count (max_num_hands = 2)
    avg_confidence = len(results.multi_hand_landmarks) / 2

    # Log low-confidence images
    if avg_confidence < 0.15:  # Threshold for low-confidence images
        low_confidence_images.append([class_name, image_path, avg_confidence])
        cv2.imwrite(os.path.join(LOW_CONFIDENCE_DIR,
                                 os.path.basename(image_path)), image)

    return [class_name, image_path, avg_confidence] + landmarks


# Dataset Processing
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


# Run Processing
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

# Save low-confidence images metadata
low_conf_df = pd.DataFrame(low_confidence_images,
                           columns=["Class", "Filename", "Confidence"])
low_conf_df.to_csv("low_confidence_images.csv", index=False)

# Final logging
print("\nHand landmark extraction complete!")
print(f"Train dataset size: {len(train_df)} images")
print(f"Test dataset size: {len(test_df)} images")
print(f"Skipped images saved in '{SKIPPED_DIR}'")
print(f"Low-confidence images saved in '{LOW_CONFIDENCE_DIR}'")

if skipped_confidences:
    print(f"🔍 Average confidence of skipped images: {np.mean(
        skipped_confidences):.4f}")
