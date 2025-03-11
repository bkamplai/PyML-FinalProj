import os
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore

# Define dataset paths
DATASET_PATH = "../dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "Train_Alphabet")
TEST_PATH = os.path.join(DATASET_PATH, "Test_Alphabet")


# Function to count images per class
def count_images_in_classes(dataset_path):
    class_counts = {}
    for class_name in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))
    return class_counts


# Count images in train and test sets
train_class_counts = count_images_in_classes(TRAIN_PATH)
test_class_counts = count_images_in_classes(TEST_PATH)

# Convert to DataFrame for better visualization
dataset_distribution = pd.DataFrame({
    "Class": list(train_class_counts.keys()),
    "Train Count": list(train_class_counts.values()),
    "Test Count": [test_class_counts.get(cls, 0) for cls in
                   train_class_counts.keys()]
})

# Display dataset distribution
print(dataset_distribution)

# Plot the class distrubutions
plt.figure(figsize=(12, 5))
plt.bar(train_class_counts.keys(), train_class_counts.values(), alpha=0.7,
        label="Train Set")
plt.bar(test_class_counts.keys(), test_class_counts.values(), alpha=0.7,
        label="Test Set")
plt.xlabel("ASL Letter Classes")
plt.ylabel("Number of Images")
plt.title("Dataset Class Distribution")
plt.xticks(rotation=45)
plt.legend()
plt.show()
plt.savefig("Screenshots/dataset_distribution.png")
