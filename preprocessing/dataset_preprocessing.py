import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns  # type: ignore
from tensorflow.keras.preprocessing.image import (  # type: ignore
    ImageDataGenerator
)
from typing import List, Tuple, Dict

# Define dataset paths
train_path: str = "../dataset/Train_Alphabet"
test_path: str = "../dataset/Test_Alphabet"

# Get class labels from folder names
classes: List[str] = sorted(os.listdir(train_path))
# print(classes)


def load_sample_images(dataset_path: str, num_samples: int = 5) -> \
        Tuple[List[np.ndarray], List[str]]:
    """Load a sample of images from teh dataset for visualization."""
    sample_images, sample_labels = [], []

    for label in classes:
        label_path = os.path.join(dataset_path, label)

        if os.path.isdir(label_path):
            images = os.listdir(label_path)[:num_samples]

            for img_name in images:
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path)

                if img is not None:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    sample_images.append(img)
                    sample_labels.append(label)

    return sample_images, sample_labels


def count_images_per_class(dataset_path: str) -> Dict[str, int]:
    """Count the number of images per class in the dataset."""
    return {label: len(os.listdir(os.path.join(dataset_path, label))) for
            label in classes}


def preprocess_image(image_path: str,
                     target_size: Tuple[int, int] = (128, 128),
                     normalize: bool = True) -> np.ndarray:
    """Load and preprocess an image by resizing and normalizing."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img / 255.0 if normalize else img


if __name__ == "__main__":
    print("Loading dataset samples...")
    sample_images, sample_labels = load_sample_images(train_path,
                                                      num_samples=2)

    # Display sample images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle("Sample Images from ASL Alphabet Dataset")

    for i, ax in enumerate(axes.flat):
        if i < len(sample_images):
            ax.imshow(sample_images[i])
            ax.set_title(sample_labels[i])
            ax.axis("off")

    plt.savefig("images/sample_images.png")
    plt.close()

    print("Counting dataset images...")
    train_counts = count_images_per_class(train_path)
    test_counts = count_images_per_class(test_path)

    # Plot class distribution and save figures
    plt.figure(figsize=(14, 6))
    sns.barplot(x=list(train_counts.keys()), y=list(train_counts.values()))
    plt.xticks(rotation=90)
    plt.xlabel("ASL Letters")
    plt.ylabel("Number of Images")
    plt.title("Training Dataset Class Distribution")
    plt.savefig("images/train_class_distribution.png")
    plt.close()

    print("Preprocessing a sample image...")
    sample_img_path = os.path.join(train_path, classes[0], os.listdir(
        os.path.join(train_path, classes[0]))[0])
    processed_img = preprocess_image(sample_img_path)

    plt.imshow(processed_img)
    plt.title(f"Preprocessed Image ({classes[0]})")
    plt.axis("off")
    plt.savefig("images/preprocessed_sample.png")
    plt.close()

    print("Applying data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.5, 1.5]
    )

    # Apply augementation to one sample image
    # # Expand dimensions for augmentation
    img_expanded = np.expand_dims(processed_img * 255, axis=0)
    augmented_images = [datagen.random_transform(img_expanded[0]) / 255.0 for
                        _ in range(5)]

    # Save augmented images
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    fig.suptitle("Augmented Image Examples")

    for i, ax in enumerate(axes.flat):
        ax.imshow(augmented_images[i])
        ax.axis("off")

    plt.savefig("images/augmented_images.png")
    plt.close()

    print("Preprocessing complete.")
    print("Check generated images in the 'images' directory.")
