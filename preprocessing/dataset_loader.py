import pickle
import tensorflow as tf  # type: ignore
from tensorflow.keras.preprocessing.image import (  # type: ignore
    ImageDataGenerator
)
from typing import Tuple

# Dataset paths
DATASET_1 = "../dataset/ASL_Alphabet_Dataset/asl_alphabet_train"
DATASET_2 = "../dataset/Train_Alphabet"
TESTSET_1 = "../dataset/ASL_Alphabet_Dataset/asl_alphabet_test"
TESTSET_2 = "../dataset/Test_Alphabet"

IMG_SIZE: Tuple[int, int] = (128, 128)
BATCH_SIZE: int = 32


def create_generators() -> Tuple[tf.data.Dataset, tf.data.Dataset,
                                 tf.data.Dataset]:
    print("Creating datasets...")

    train_ds_1 = tf.keras.utils.image_dataset_from_directory(
        DATASET_1,
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
