import tensorflow as tf

# Load the trained model
model_path = "Models/asl_fingerspell_mobilenet_finetuned.keras"
model = tf.keras.models.load_model(model_path)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Enable model optimization
tflite_model = converter.convert()
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Save the converted TFLite model
tflite_model_path = "Models/asl_fingerspell_mobilenet_finetuned.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)


print(f"Model converted to TFLite format and saved to {tflite_model_path}")