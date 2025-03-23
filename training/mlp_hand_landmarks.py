import pickle
import pandas as pd  # type: ignore
import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam    # type: ignore
from tensorflow.keras.regularizers import l2    # type: ignore
from tensorflow.keras.callbacks import (    # type: ignore
    EarlyStopping, ReduceLROnPlateau)
from sklearn.model_selection import train_test_split    # type: ignore
from sklearn.preprocessing import LabelEncoder, OneHotEncoder   # type: ignore

# Load dataset
train_df = pd.read_csv("../processed_data/Train_Data__No_Edge_Features.csv")
test_df = pd.read_csv("../processed_data/test_hand_landmarks_no_edge.csv")

# Separate features and labels
X_train = train_df.iloc[:, 2:].values
y_train = train_df.iloc[:, 0].values    # Class labels

X_test = test_df.iloc[:, 2:].values
y_test = test_df.iloc[:, 0].values

# Encode labels (A-Z, Blank -> 0-26)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# One-hot encode labels
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = one_hot_encoder.fit_transform(y_train_encoded.reshape(-1, 1))
y_test_onehot = one_hot_encoder.transform(y_test_encoded.reshape(-1, 1))

# Split training into train/validation (80/20 split)
X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(
    X_train, y_train_onehot, test_size=0.2, random_state=42)

# Normalize input features
X_train, X_val, X_test = X_train / np.max(X_train), X_val / np.max(X_train), \
    X_test / np.max(X_train)

# Define MLP model with improved architecture
model = Sequential([
    Dense(256, activation='relu', kernel_regularizer=l2(0.001), input_shape=(
        X_train.shape[1],)),
    Dropout(0.2),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(y_train_onehot.shape[1], activation='softmax')  # Output layer
])

# Compile model with updated learning rate
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [EarlyStopping(patience=5, restore_best_weights=True),
             ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)]

# Train model with increased batch size
history = model.fit(X_train, y_train_onehot, epochs=50, batch_size=64,
                    validation_data=(X_val, y_val_onehot), callbacks=callbacks)

# Save history to a PKL file
with open("Training History/training_history_mlp_tuned.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test_onehot)
print(f"Test Accuracy: {test_acc:.4f}")

# Save model
model.save("Models/mlp_hand_landmark_model_tuned.keras")
