import os
import cv2
import mediapipe as mp  # type: ignore
import pandas as pd  # type: ignore
from tqdm import tqdm

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define dataset paths
DATASET_PATH = "../dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "Train_Alphabet")
TEST_PATH = os.path.join(DATASET_PATH, "Test_Alphabet")

# Set up the Hands model
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                       min_detection_confidence=0.5)


def extract_landmarks(image_path: str):
    """Extract 21 hand landmarks from an image using MediaPipe Hands."""
    image = cv2.imread(image_path)
    if image is None:
        return None  # Image could not be loaded

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        landmarks = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])  # Extract X, Y, Z

        # Ensure exactly 63 landmark values (21 landmarks * 3 coordinates)
        if len(landmarks) == 63:
            return landmarks
        else:
            print(f"Warning: {image_path} has {len(landmarks)} landmarks " +
                  "instead of 63. Skipping.")
            print(f"Extracted {len(landmarks)} landmarks for {image_path}")

    return None  # No hand detected or incorrect landmarks


def process_dataset(dataset_path: str, dataset_type: str):
    """Process images in dataset to extract hand landmarks."""
    data = []

    for class_name in tqdm(sorted(os.listdir(dataset_path)),
                           desc=f"Processing {dataset_type}"):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            landmarks = extract_landmarks(img_path)

            if landmarks:
                data.append([class_name] + landmarks)

    return data


# Extract landmarks from train and test sets
train_data = process_dataset(TRAIN_PATH, "Train")
test_data = process_dataset(TEST_PATH, "Test")

# Debugging: Check if rows have the correct number of elements
for i, row in enumerate(train_data):
    if len(row) != 65:  # Expecting 65 columns (63 landmarks + class + filename)
        print(f"⚠️ Row {i} has {len(row)} values instead of 65: {row}")

# Convert to DataFrame
columns = ["Class", "Filename"] + [f"LM_{i}_{axis}" for i in range(21) for
                                   axis in ["x", "y", "z"]]
train_df = pd.DataFrame(train_data, columns=columns)
test_df = pd.DataFrame(test_data, columns=columns)

# Save extracted landmarks
train_df.to_csv("train_hand_landmarks.csv", index=False)
test_df.to_csv("test_hand_landmarks.csv", index=False)

print(f"Hand landmark extraction complete! Train: {len(train_df)}, Test: {
    len(test_df)}")
