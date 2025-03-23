import os
import cv2  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

# Define dataset paths
DATASET_PATH = "../dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "Train_Alphabet")
TEST_PATH = os.path.join(DATASET_PATH, "Test_Alphabet")

# Define blur threshold (lower = more strict, higher = lenient)
LAPLACIAN_THRESHOLD = 130.0  # Higher value to reduce false positives
EDGE_THRESHOLD = 45  # Ensures edges exist
HISTOGRAM_THRESHOLD = 9.0  # Minimum contrast level


def compute_laplacian_variance(image: np.ndarray) -> float:
    """ Compute Laplacian variance for sharpness detection. """
    return cv2.Laplacian(image, cv2.CV_64F).var()


def compute_edge_density(image: np.ndarray) -> float:
    """ Compute the density of detected edges using Canny edge detection. """
    edges = cv2.Canny(image, 100, 200)
    return np.count_nonzero(edges) / edges.size


def compute_histogram_sharpness(image: np.ndarray) -> float:
    """ Compute contrast using histogram spread of pixel intensities. """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist.std()


def is_blurry(image_path: str) -> bool:
    """ Analyze an image with multiple blur detection methods. """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian_score = compute_laplacian_variance(img)
    edge_density = compute_edge_density(img)
    histogram_sharpness = compute_histogram_sharpness(img)

    # Image is blurry if it fails all three tests
    return (laplacian_score < LAPLACIAN_THRESHOLD and
            edge_density < EDGE_THRESHOLD and
            histogram_sharpness < HISTOGRAM_THRESHOLD)


def detect_blurry_images(dataset_path: str):
    """ Scan dataset for blurry images using improved detection methods. """
    blurry_images = []
    for class_name in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if is_blurry(img_path):
                blurry_images.append((class_name, img_name))

    return blurry_images


# Run blur detection on train and test datasets
blurry_train_images = detect_blurry_images(TRAIN_PATH)
blurry_test_images = detect_blurry_images(TEST_PATH)

# Convert to DataFrame for better visualization
blurry_df = pd.DataFrame(blurry_train_images + blurry_test_images,
                         columns=["Class", "Filename"])

# Save results
blurry_df.to_csv("blurry_images.csv", index=False)
print("Blurry images detected! Check 'blurry_images.csv' for details.")
