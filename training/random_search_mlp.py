import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

# Load dataset from CSV
train_df = pd.read_csv("train_hand_landmarks.csv")
test_df = pd.read_csv("test_hand_landmarks.csv")

# Separate features and labels
X_train = train_df.iloc[:, 2:].values  # Skip class & filename
y_train = train_df.iloc[:, 0].values   # Class labels

X_test = test_df.iloc[:, 2:].values
y_test = test_df.iloc[:, 0].values

# Split training into train/validation (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Encode labels after the split
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# One-hot encode labels
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = one_hot_encoder.fit_transform(y_train_encoded.reshape(-1, 1))
y_val_onehot = one_hot_encoder.transform(y_val_encoded.reshape(-1, 1))

# Normalize input features
X_train, X_val, X_test = X_train / np.max(X_train), X_val / np.max(X_train), \
    X_test / np.max(X_train)

# Function to create MLP model
def create_mlp_model(num_neurons=128, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        Dense(num_neurons, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(num_neurons, activation="relu"),
        Dropout(dropout_rate),
        Dense(y_train_onehot.shape[1], activation="softmax")  # Output layer
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Wrap Keras model with scikit-learn wrapper
mlp_model = KerasClassifier(
    model=create_mlp_model,
    epochs=20,
    batch_size=32,
    verbose=0
)

# Define hyperparameter space
param_dist = {
    "model__num_neurons": [64, 128, 256],
    "model__dropout_rate": [0.1, 0.2, 0.3, 0.4],
    "model__learning_rate": [0.001, 0.0005, 0.0001],
    "batch_size": [32, 64, 128],
    "epochs": [20, 30, 50]
}

random_search = RandomizedSearchCV(
    estimator=mlp_model,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    verbose=2,
    n_jobs=-1,
    scoring="accuracy"
)

print(f"X_train shape: {X_train.shape}")
print(f"y_train_encoded shape: {y_train_encoded.shape}")

random_search.fit(X_train, y_train_encoded)

# Display best parameters
print("Best Hyperparameters:", random_search.best_params_)

# Train final model with best parameters
best_params = random_search.best_params_
final_model = create_mlp_model(
    num_neurons=best_params["model__num_neurons"],
    dropout_rate=best_params["model__dropout_rate"],
    learning_rate=best_params["model__learning_rate"]
)

history = final_model.fit(
    X_train, y_train_onehot,
    validation_data=(X_val, y_val_onehot),
    batch_size=best_params["batch_size"],
    epochs=best_params["epochs"],
    verbose=1
)

# Save final model
final_model.save("Models/mlp_hand_landmarks_tuned.keras")

# Save training history for visualization
with open("Training History/training_history_mlp_tuned.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("Final model trained and saved!")