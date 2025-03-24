import cv2
import numpy as np
import tensorflow as tf

def main():
    # Load the model
    model = tf.keras.models.load_model('../training/Models/asl_fingerspell_mobilenet_finetuned.keras')

    # All categories for the ASL alphabet
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['Blank']

    # Start the webcam or video capture
    cap = cv2.VideoCapture('alphabet.mp4')  # Change to 0 for webcam capture

    if not cap.isOpened():
        print("Can't load video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:  # If no more frames, exit the loop
            print("Failed to capture frame. Exiting.")
            break

        # Check the shape of the frame (debugging purpose)
        print(f"Original Frame Shape: {frame.shape}")

        # Convert from BGR (OpenCV format) to RGB (TensorFlow format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame to match the model's expected input size (128x128)
        frame_resized = cv2.resize(frame_rgb, (128, 128))  # Resize to 128x128
        frame_resized = np.expand_dims(frame_resized, axis=0)
        frame_resized = frame_resized / 255.0  # Normalize the image for prediction

        # Make prediction
        prediction = model.predict(frame_resized)
        letter = labels[np.argmax(prediction)]  # Get the predicted letter

        # Display the predicted letter on the original frame (in BGR format)
        cv2.putText(frame, f'Predicted letter for sign: {letter}', 
                    (10, 30),  # Position the text at the top-left corner
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the frame with the prediction text (using original frame, in BGR format)
        cv2.imshow('ASL Signsense', frame)

        # Close if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
