import pickle
import matplotlib.pyplot as plt


# Load training history from a PKL file
def load_history(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


# Function to plot training curves
def plot_training_curves(history, title="MLP Training Curves"):
    epochs = range(1, len(history["accuracy"]) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["accuracy"], "b", label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"], "r", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["loss"], "b", label="Training Loss")
    plt.plot(epochs, history["val_loss"], "r", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("../training/Screenshots/mlp_training_curves_tuned.png")
    plt.show()


# Load history and plot
history_file = "../training/Training History/training_history_mlp_tuned.pkl"
history_data = load_history(history_file)
plot_training_curves(history_data, title="MLP Hand Landmarks Tuned Training")
