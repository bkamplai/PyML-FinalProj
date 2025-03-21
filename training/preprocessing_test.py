import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Define dataset base path
DATASET_BASE_PATH = "../dataset"

# Load low-confidence images CSV
csv_file = "low_confidence_images.csv"
df = pd.read_csv(csv_file)


# Ensure the dataset path is correctly referenced
def get_correct_path(csv_filename):
    """Convert relative path from CSV to absolute dataset path"""
    if not os.path.exists(csv_filename):  # If path doesn't exist, fix it
        corrected_path = os.path.join(DATASET_BASE_PATH,
                                      *csv_filename.split("/")[-3:])
        if os.path.exists(corrected_path):
            return corrected_path
    return csv_filename  # Return original if it already exists


# Apply path correction to the DataFrame
df["Corrected_Filename"] = df["Filename"].apply(get_correct_path)

# Sample images for visualization
# Random 10 low-confidence images
sample_images = df.sample(n=10, random_state=42)


def preprocess_image(image_path):
    """Apply preprocessing steps to improve contrast and clarity"""
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)

    # Convert back to RGB for visualization
    processed_image = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)

    return processed_image


# Visualize original and processed images
fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(10, 25))

for i, (_, row) in enumerate(sample_images.iterrows()):
    img_path = row["Corrected_Filename"]

    if not os.path.exists(img_path):
        print(f"Missing file: {img_path}")  # Log missing files
        continue

    original = cv2.imread(img_path)
    processed = preprocess_image(img_path)

    if original is None or processed is None:
        continue

    # Convert to RGB (OpenCV loads in BGR format)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    axes[i, 0].imshow(original)
    axes[i, 0].set_title(f"Original: {row['Class']}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(processed)
    axes[i, 1].set_title(f"Processed: {row['Class']}")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig("Screenshots/low_confidence_analysis.png")  # Save for reference
plt.show()

print("Preprocessing analysis complete! Check " +
      "'Screenshots/low_confidence_analysis.png'.")
