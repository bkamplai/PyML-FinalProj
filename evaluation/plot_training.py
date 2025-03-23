import pickle   # type: ignore
import matplotlib.pyplot as plt

# Load training history from the saved model
cnn_history_path = "training_history_improved.pkl"
mobilenet_history_path = "training_history_mobilenet.pkl"
mobilenet_finetuned_history_path = "training_history_mobilenet_finetuned.pkl"


def load_training_history(file_path: str) -> dict:
    """
    Loads the training history from a saved picked file.
    Args:
        file_path (str): Path to the training history file.
    Returns:
        history (dict): Dictionary containing loss & accuracy values for each
        epoch.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def plot_training_curves(history):
    """
    Plots training & validation accuracy and loss over epochs.
    Args:
        history (dict): Dictionary containing 'loss', 'val_loss', 'accuracy',
        'val_accuracy'.
    """
    epochs = range(1, len(history["accuracy"]) + 1)

    # Plot Accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["accuracy"], "b-", label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"],
             "r-", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs. Validation Accuracy")
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["loss"], "b-", label="Training Loss")
    plt.plot(epochs, history["val_loss"], "r-", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves_mobilenet_finetuned.png")  # Save the figure
    plt.show()


def plot_comparison(cnn_history: dict, mobilenet_history: dict,
                    mobilenet_finetuned_history: dict):
    """
    Plots training & validation accuracy/loss comparisons between CNN,
    MobileNetV2 and Fine-Tuned MobileNetV2.

    Args:
        cnn_history (dict): History of CNN model training.
        mobilenet_history (dict): History of MobileNetV2 training.
        mobilenet_finetuned_history (dict): History of MobileNetV2 fine-tuned.
    """
    epochs_cnn = range(1, len(cnn_history["accuracy"]) + 1)
    epochs_mobilenet = range(1, len(mobilenet_history["accuracy"]) + 1)
    epochs_mobilenet_finetuned = range(1, len(mobilenet_finetuned_history[
        "accuracy"]) + 1)

    # Set up the figure
    plt.figure(figsize=(12, 5))

    # Plot Accuracy Comparison
    plt.subplot(1, 2, 1)
    plt.plot(epochs_cnn, cnn_history["accuracy"], "b-", label="CNN Train Acc")
    plt.plot(epochs_cnn, cnn_history["val_accuracy"], "b--",
             label="CNN Val Acc")
    plt.plot(epochs_mobilenet, mobilenet_history["accuracy"], "r-",
             label="MobileNet Train Acc")
    plt.plot(epochs_mobilenet, mobilenet_history["val_accuracy"], "r--",
             label="MobileNet Val Acc")
    plt.plot(epochs_mobilenet_finetuned,
             mobilenet_finetuned_history["accuracy"], "g-",
             label="Fine-Tuned MobileNet Train Acc")
    plt.plot(epochs_mobilenet_finetuned,
             mobilenet_finetuned_history["val_accuracy"], "g--",
             label="Fine-Tuned MobileNet Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("CNN vs. MobileNetV2 vs. Fine-Tuned MobileNet - Accuracy")
    plt.legend()

    # Plot Loss Comparison
    plt.subplot(1, 2, 2)
    plt.plot(epochs_cnn, cnn_history["loss"], "b-", label="CNN Train Loss")
    plt.plot(epochs_cnn, cnn_history["val_loss"], "b--", label="CNN Val Loss")
    plt.plot(epochs_mobilenet, mobilenet_history["loss"], "r-",
             label="MobileNet Train Loss")
    plt.plot(epochs_mobilenet, mobilenet_history["val_loss"], "r--",
             label="MobileNet Val Loss")
    plt.plot(epochs_mobilenet_finetuned, mobilenet_finetuned_history["loss"],
             "g-", label="Fine-Tuned MobileNet Train Loss")
    plt.plot(epochs_mobilenet_finetuned,
             mobilenet_finetuned_history["val_loss"], "g--",
             label="Fine-Tuned MobileNet Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("CNN vs. MobileNetV2 vs. Fine-Tuned MobileNet - Loss")
    plt.legend()

    # Save and Show
    plt.tight_layout()
    plt.savefig("cnn_vs_mobilenet_finetuned.png")
    plt.show()


if __name__ == "__main__":
    print("Loading training histories...")
    cnn_history = load_training_history(cnn_history_path)
    mobilenet_history = load_training_history(mobilenet_history_path)
    mobilenet_finetuned_history = load_training_history(
        mobilenet_finetuned_history_path)

    print("Plotting training curves comparison...")
    plot_comparison(cnn_history, mobilenet_history,
                    mobilenet_finetuned_history)
    print("Comparison plot saved as 'cnn_vs_mobilenet_finetuned.png'.")
