import os
import random
import base64
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model('model/model.DenseNet121.keras')

# RAF-DB emotion labels
emotion_labels = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

# Load emoji image paths
emoji_paths = {
    emotion: [f'static/emojis/{emotion}/{img}' for img in os.listdir(f'static/emojis/{emotion}')]
    for emotion in emotion_labels
}

# Initialize webcam
camera = cv2.VideoCapture(0)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect and crop face
def detect_and_crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        return frame[y:y+h, x:x+w]
    return None  # No face detected

# Preprocess frame
def preprocess_frame(frame):
    resized = cv2.resize(frame, (100, 100))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    processed = preprocess_input(np.expand_dims(rgb, axis=0))  # Preprocessed for DenseNet
    return processed

# Stream video with predictions
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Crop face if found
            cropped = detect_and_crop_face(frame)
            input_frame = cropped if cropped is not None else frame

            preprocessed = preprocess_frame(input_frame)
            prediction = model.predict(preprocessed, verbose=0)
            emotion_index = np.argmax(prediction)
            emotion = emotion_labels[emotion_index]

            # Draw emotion label
            cv2.putText(frame, f'Emotion: {emotion}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict_picture', methods=['POST'])
def predict_picture():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)

    # Decode and convert to OpenCV image (BGR)
    np_img = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Convert BGR to RGB (as model is trained on RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize and normalize
    resized = cv2.resize(img_rgb, (100, 100))
    normalized = resized / 255.0
    image_np = np.expand_dims(normalized, axis=0)

    prediction = model.predict(image_np, verbose=0)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]

    emoji_list = emoji_paths.get(emotion, [])
    emoji_path = '/' + random.choice(emoji_list) if emoji_list else ''

    return jsonify({'emotion': emotion, 'emoji_path': emoji_path})

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    # Check if the request contains a file
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']

    # Read the uploaded image file into memory
    in_memory_file = BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)  # Decode image using OpenCV

    # Convert BGR (OpenCV default) to RGB (for model and display)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize image to match model input size
    resized = cv2.resize(img_rgb, (100, 100))

    # Normalize image to [0,1] range
    normalized = resized / 255.0

    # Add batch dimension (1, 100, 100, 3)
    image_np = np.expand_dims(normalized, axis=0)

    # Make prediction using the trained model
    prediction = model.predict(image_np, verbose=0)
    emotion_index = np.argmax(prediction[0])  # Get index of highest probability
    emotion = emotion_labels[emotion_index]  # Map to emotion label

    # Print prediction details to console (optional debug)
    print("Prediction:", prediction)
    print("Emotion Index:", emotion_index)
    print("Emotion:", emotion)

    # Get corresponding emoji path list and select one at random
    emoji_list = emoji_paths.get(emotion, [])
    emoji_path = '/' + random.choice(emoji_list) if emoji_list else ''

    # Convert image to base64 to render in HTML without saving to disk
    img_pil = Image.fromarray(img_rgb)
    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    # Render result on upload.html page with emotion, emoji, and uploaded image
    return render_template('upload.html', emotion=emotion, emoji_path=emoji_path, uploaded_image=img_base64)

if __name__ == '__main__':
    app.run(debug=True)