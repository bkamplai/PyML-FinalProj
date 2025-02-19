import pickle   # type: ignore
import matplotlib.pyplot as plt

# Load training history from the saved model
history_path = "training_history_improved.pkl"


def load_training_history():
    """
    Loads the training history from a saved picked file.
    Returns:
        history (dict): Dictionary containing loss & accuracy values for each
        epoch.
    """
    with open(history_path, "rb") as f:
        history = pickle.load(f)
    return history


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
    plt.savefig("training_curves_improved.png")  # Save the figure
    plt.show()


if __name__ == "__main__":
    print("Loading training history...")
    history = load_training_history()
    plot_training_curves(history)
    print("Training curves saved as 'training_curves_improved.png'.")
