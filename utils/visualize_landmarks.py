import os
import cv2
import mediapipe as mp
import pandas as pd
import random

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define dataset path
DATASET_PATH = "../dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "Train_Alphabet")

# Load extracted landmarks
landmark_df = pd.read_csv("train_hand_landmarks.csv")

# Set up MediaPipe Hands for visualization
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                       min_detection_confidence=0.5)


def visualize_landmarks(image_path: str, landmarks: list):
    """ Overlay hand landmarks onto an image. """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load {image_path}")
        return

    height, width, _ = image.shape

    for i in range(0, len(landmarks), 3):   # Each landmark has (x, y, z)
        x = int(landmarks[i] * width)
        y = int(landmarks[i + 1] * height)
        # Green dots for landmarks
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Hand Landmark Visualization", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Pick random images for visualization
sample_images = random.sample(list(landmark_df.itertuples(index=False)), 5)

for sample in sample_images:
    class_name, filename, *landmarks = sample
    img_path = os.path.join(TRAIN_PATH, class_name, filename)
    visualize_landmarks(img_path, landmarks)

print("Landmark visualization complete! Close the images to continue.")
