import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Paths
SKIPPED_DIR = "skipped_images"
OUTPUT_CSV = "skipped_images_analysis.csv"
SAMPLE_DIR = "skipped_samples"
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Store analysis results
skipped_counts = Counter()
brightness_levels = []
contrast_levels = []
sample_images = []


# Function to compute brightness
def compute_brightness(image):
    return np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


# Function to compute contrast
def compute_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std()


# Process skipped images
for filename in os.listdir(SKIPPED_DIR):
    image_path = os.path.join(SKIPPED_DIR, filename)
    image = cv2.imread(image_path)

    if image is None:
        continue  # Skip unreadable images

    # Extract class from filename (assuming format "Class_XXXXX.png")
    class_name = filename.split("_")[0] if "_" in filename else "Unknown"
    skipped_counts[class_name] += 1

    # Compute brightness & contrast
    brightness = compute_brightness(image)
    contrast = compute_contrast(image)
    brightness_levels.append(brightness)
    contrast_levels.append(contrast)

    # Save a few sample images for manual review
    if len(sample_images) < 10:
        sample_path = os.path.join(SAMPLE_DIR, filename)
        cv2.imwrite(sample_path, image)
        sample_images.append(sample_path)

# Save skipped counts to CSV
df = pd.DataFrame.from_dict(skipped_counts, orient='index',
                            columns=['Skipped Count'])
df = df.sort_values(by='Skipped Count', ascending=False)
df.to_csv(OUTPUT_CSV)

# Plot histogram of brightness & contrast
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(brightness_levels, bins=20, color='blue', alpha=0.7)
plt.xlabel("Brightness")
plt.ylabel("Frequency")
plt.title("Brightness Distribution in Skipped Images")

plt.subplot(1, 2, 2)
plt.hist(contrast_levels, bins=20, color='red', alpha=0.7)
plt.xlabel("Contrast")
plt.ylabel("Frequency")
plt.title("Contrast Distribution in Skipped Images")

plt.tight_layout()
plt.savefig("./Screenshots/skipped_images_analysis.png")
plt.show()

# Print summary
print("\n🔍 Skipped Images Analysis Complete!")
print(f"📊 Skipped classes breakdown saved to: {OUTPUT_CSV}")
print(f"🖼️ Sample images saved in: {SAMPLE_DIR}")
print("📉 Brightness & contrast analysis saved to: " +
      "./Screenshots/skipped_images_analysis.png")
