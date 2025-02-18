import os
import pickle
import tensorflow as tf  # type: ignore
from tensorflow.keras.preprocessing.image import (  # type: ignore
    ImageDataGenerator
)
from typing import Tuple, Dict


# Define dataset paths
DATASET_PATH: str = "../dataset"
TRAIN_PATH: str = os.path.join(DATASET_PATH, "Train_Alphabet")
TEST_PATH: str = os.path.join(DATASET_PATH, "Test_Alphabet")

# Target image size and batch size
IMG_SIZE: Tuple[int, int] = (128, 128)  # (Width, Height)
BATCH_SIZE: int = 32  # Adjust based on available memory

# Define class labels
CLASSES: Tuple[str, ...] = tuple(
    sorted(os.listdir(TRAIN_PATH)))  # Read class names


def create_generators() -> Tuple[
    tf.keras.preprocessing.image.DirectoryIterator,
    tf.keras.preprocessing.image.DirectoryIterator,
    tf.keras.preprocessing.image.DirectoryIterator,
]:
    """
    Creates training, validation, and testing data generators to efficiently
    load images.

    Returns:
        Tuple containing:
            - train_generator (DirectoryIterator): Augmented training data
            - val_generator (DirectoryIterator): Validation data (no
                             augmentation)
            - test_generator (DirectoryIterator): Test data (no augmentation)
    """

    # Data augmentation for training
    train_datagen: ImageDataGenerator = ImageDataGenerator(
        rescale=1.0 / 255,  # Normalize images to [0, 1] range
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.5, 1.5],
        # Automatically split into 80% train, 20% validation
        validation_split=0.2,
    )

    test_datagen: ImageDataGenerator = ImageDataGenerator(
        rescale=1.0 / 255)  # Only rescale test set

    # Load training & validation data using generator
    train_generator: tf.keras.preprocessing.image.DirectoryIterator = \
        train_datagen.flow_from_directory(
            TRAIN_PATH,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="sparse",  # Sparse labels for classification
            subset="training",
        )

    val_generator: tf.keras.preprocessing.image.DirectoryIterator = \
        train_datagen.flow_from_directory(
            TRAIN_PATH,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="sparse",
            subset="validation",
        )

    # Load test data (no augmentation)
    test_generator: tf.keras.preprocessing.image.DirectoryIterator = \
        test_datagen.flow_from_directory(
            TEST_PATH,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="sparse",
        )

    return train_generator, val_generator, test_generator


if __name__ == "__main__":
    print("Creating data generators...")

    train_generator, val_generator, test_generator = create_generators()

    # Save label mapping for future reference
    label_to_index: Dict[str, int] = train_generator.class_indices
    with open("label_mapping.pkl", "wb") as f:
        pickle.dump(label_to_index, f)

    print(f"Class labels saved: {label_to_index}")
    print("Data generators are ready. Use train_generator, val_generator, and "
          + "test_generator for training.")
