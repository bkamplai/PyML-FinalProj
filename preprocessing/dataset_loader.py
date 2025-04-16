import pickle
import tensorflow as tf  # type: ignore
from tensorflow.keras.preprocessing.image import (  # type: ignore
    ImageDataGenerator
)
from typing import Tuple

# Define dataset paths
DATASET_PATH: str = "../dataset/Combined Dataset"
TRAIN_PATH: str = os.path.join(DATASET_PATH, "Train")
TEST_PATH: str = os.path.join(DATASET_PATH, "Test")

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
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    train_ds_2 = tf.keras.utils.image_dataset_from_directory(
        DATASET_2,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds_1 = tf.keras.utils.image_dataset_from_directory(
        DATASET_1,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds_2 = tf.keras.utils.image_dataset_from_directory(
        DATASET_2,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    test_ds_1 = tf.keras.utils.image_dataset_from_directory(
        TESTSET_1,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    test_ds_2 = tf.keras.utils.image_dataset_from_directory(
        TESTSET_2,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    train_ds = train_ds_1.concatenate(train_ds_2).prefetch(
        buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds_1.concatenate(val_ds_2).prefetch(
        buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds_1.concatenate(test_ds_2).prefetch(
        buffer_size=tf.data.AUTOTUNE)

    # Use a temporary generator to extract class names
    temp_gen = ImageDataGenerator().flow_from_directory(
        DATASET_1,
        target_size=IMG_SIZE,
        batch_size=1,
        shuffle=False
    )

    with open("label_mapping.pkl", "wb") as f:
        pickle.dump(temp_gen.class_indices, f)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = create_generators()
    print("Datasets created and label mapping saved.")
