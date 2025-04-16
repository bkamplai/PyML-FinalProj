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
labels = ['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# For uploading video
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov', 'avi'}

# Default image/video path
uploaded_image_path = None
uploaded_video_path = None

# Handle webcam
@app.route('/webcam', methods=['GET'])
def webcam():
    return render_template('webcam.html')

# Run the webcam frames through opencv and give prediction.
@app.route('/webcam_feed')
def webcam_feed():
    cap = cv2.VideoCapture(0)
    # https://docs.opencv.org/3.4/dd/d00/tutorial_js_video_display.html
    # https://pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/
    
    if not cap.isOpened():
        return jsonify({'error': 'Can\'t load webcam'})

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

        # Display letter prediction on top left (white and black)
        cv2.putText(frame, f"Predicted signed letter: {letter}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Confidence: {prediction[0][letter_prediction] * 100:.2f}%", (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Predicted signed letter: {letter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Confidence: {prediction[0][letter_prediction] * 100:.2f}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        # Convert frame to JPEG and use as response
        _, jpeg = cv2.imencode('.jpg', frame)
        if jpeg is not None:
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    
    cap.release()


# Handle uploads
@app.route('/upload_image', methods=['POST'])
def upload_image():
    global uploaded_image_path

    if 'image' not in request.files:
        return jsonify({'error': 'No image file'})
    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if image:
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        uploaded_image_path = image_path

        # Image sucessfully uploaded
        return jsonify({'message': "Image uploaded"})
    else:
        return jsonify({'error': 'Incorrect file format'})

# Have user upload video
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
@app.route('/predict_video')
def predict_video():
    global uploaded_video_path
    if uploaded_video_path is None:
        return jsonify({'error': 'No video uploaded'})
    return Response(predict_letters(uploaded_video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run image through model
@app.route('/predict_image', methods=['GET'])
def predict_image():
    global uploaded_image_path
    if uploaded_image_path is None:
        return jsonify({'error': 'No image uploaded'})

    # Prepare the image
    image = cv2.imread(uploaded_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (128, 128))
    image_resized = image_resized / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)

    # Predict letter in image
    prediction = model.predict(image_resized)
    letter_prediction = np.argmax(prediction)
    letter = labels[letter_prediction]
    confidence = prediction[0][letter_prediction]

    # White and black text
    cv2.putText(image, f"Predicted signed letter: {letter}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Confidence: {prediction[0][letter_prediction] * 100:.2f}%", (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Predicted signed letter: {letter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"Confidence: {prediction[0][letter_prediction] * 100:.2f}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert back to RGB for proper display in browser
    _, jpeg = cv2.imencode('.jpg', image_rgb)

    if jpeg is not None:
        # Convert image to byte stream for transmission
        image_bytes = jpeg.tobytes()
        os.remove(uploaded_image_path)
        uploaded_image_path = None
        return Response(image_bytes, content_type='image/jpeg')
    else:
        return jsonify({'error': 'Failed to process image'})

# Function to take in video and run frames through the model
def predict_letters(video_path):
    # Start the webcam (uploading video for now)
    cap = cv2.VideoCapture(video_path)  # Change to 0 for webcam capture

    if not cap.isOpened():
        print("Can't load video.")
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

        # Display letter prediction on top left (white and black)
        cv2.putText(frame, f"Predicted signed letter: {letter}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Confidence: {prediction[0][letter_prediction] * 100:.2f}%", (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Predicted signed letter: {letter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Confidence: {prediction[0][letter_prediction] * 100:.2f}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        # Convert frame to JPEG and use as response
        _, jpeg = cv2.imencode('.jpg', frame)
        if jpeg is not None:
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    
    cap.release()
    os.remove(uploaded_video_path)
    uploaded_video_path = None

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
