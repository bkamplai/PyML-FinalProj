import os
import cv2
import hashlib
import pandas as pd  # type: ignore
from collections import defaultdict

# Define dataset paths
DATASET_PATH = "../dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "Train_Alphabet")
TEST_PATH = os.path.join(DATASET_PATH, "Test_Alphabet")


def get_image_hash(image_path: str):
    """ Compute hash of an image for duplicate detection. """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None  # Mark as corrupted
        return hashlib.md5(img.tobytes()).hexdigest()
    except Exception:
        return None  # If any error occurs, consider corrupted


def check_dataset_for_issues(dataset_path: str):
    """ Scan dataset for duplicate or corrupted images. """
    hashes = defaultdict(list)
    corrupted_images = []

    for class_name in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img_hash = get_image_hash(img_path)

            if img_hash is None:
                corrupted_images.append((class_name, img_name))
            else:
                hashes[img_hash].append((class_name, img_name))

    # Find duplicate images (hashes with multiple occurrences)
    duplicate_images = {h: v for h, v in hashes.items() if len(v) > 1}

    return duplicate_images, corrupted_images


# Run checks on train and test sets
dup_train, corrupt_train = check_dataset_for_issues(TRAIN_PATH)
dup_test, corrupt_test = check_dataset_for_issues(TEST_PATH)

# Convert results to DataFrames
duplicate_df = pd.DataFrame([(h, len(v), v) for h, v in dup_train.items()] +
                            [(h, len(v), v) for h, v in dup_test.items()],
                            columns=["Hash", "Count", "Image Paths"])

corrupted_df = pd.DataFrame(corrupt_train + corrupt_test,
                            columns=["Class", "Filename"])

# Save results
duplicate_df.to_csv("duplicate_images.csv", index=False)
corrupted_df.to_csv("corrupted_images.csv", index=False)

print(f"Duplicate image check complete! Found {len(duplicate_df)} duplicates.")
print(f"Corrupted image check complete! Found {len(corrupted_df)} corrupted " +
      "images.")
