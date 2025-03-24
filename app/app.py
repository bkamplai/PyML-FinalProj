import cv2
import numpy as np
import tensorflow as tf
import os
# Flask imports
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
# http://127.0.0.1:5000/

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('../training/Models/asl_fingerspell_mobilenet_finetuned.keras')
# All categories for the ASL alphabet
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['Blank']

# For uploading video
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov', 'avi'}

# Default video path
uploaded_video_path = None

# Handle uploads
@app.route('/upload_video', methods=['POST'])
def upload_video():
    global uploaded_video_path
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'})
    video = request.files['video']

    if video.filename== '':
        return jsonify({'error': 'No file selected'})
    
    if video:
        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)

        uploaded_video_path = video_path

        # Video sucessfully uploaded
        return jsonify({'message': "Video uploaded"})
    else:
        return jsonify({'error': 'Incorrect file format'})

# Route to play the video
@app.route('/video_stream')
def video_stream():
    global uploaded_video_path
    if uploaded_video_path is None:
        return jsonify({'error': 'No video uploaded'})
    return Response(predict_letters(uploaded_video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to take in video and run frames through the model
def predict_letters(video_path):
    # Start the webcam (uploading video for now)
    cap = cv2.VideoCapture(video_path)  # Change to 0 for webcam capture

    if not cap.isOpened():
        print("Can't load webcam/video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:  # If no more frames break
            break

        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame for mobilenet (128x128)
        frame_resized = cv2.resize(frame_rgb, (128, 128))
        frame_resized = frame_resized / 255.0
        frame_resized = np.expand_dims(frame_resized, axis=0)

        # Make prediction
        prediction = model.predict(frame_resized)
        letter_prediction = np.argmax(prediction)
        letter = labels[letter_prediction]  # Get the predicted letter
        # predictions.append(letter)

        # Display letter prediction on top left
        cv2.putText(frame, f"Predicted signed letter: {letter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Confidence: {prediction[0][letter_prediction]}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Convert frame to JPEG and use as response
        _, jpeg = cv2.imencode('.jpg', frame)
        if jpeg is not None:
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
