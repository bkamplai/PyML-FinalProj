import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

# Load model
model = load_model(
    "../training/Models/asl_fingerspell_mobilenet_finetuned.keras"
)
print("Model loaded successfully")
# model.summary()
# print("Model input shape:", model.input_shape)
class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["Blank"]

# Tkinter setup
window = tk.Tk()
window.title("ASL Classifier")

# Webcam + canvas
video_label = Label(window)
video_label.pack()

# Prediction label
prediction_label = tk.Label(window, text="test", font=("Helvetica", 64), fg="white", bg="black")
prediction_label.pack()
prediction_label.pack(pady=10)

cap = cv2.VideoCapture(0)


def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame, (128, 128))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)


def update_frame():
    ret, frame = cap.read()

    if not ret:
        print("⚠️ Frame not captured")
        return

    frame = cv2.flip(frame, 1)  # Mirror view

    prediction = model.predict(preprocess_frame(frame), verbose=0)
    class_index = np.argmax(prediction)
    predicted_label = class_names[class_index]
    confidence = np.max(prediction)

    text = f"{predicted_label} ({confidence * 100:.2f}%)"
    prediction_label.config(text=text)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    window.after(10, update_frame)


update_frame()
window.mainloop()
cap.release()
cv2.destroyAllWindows()
