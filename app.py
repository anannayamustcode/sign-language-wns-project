from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import json
from collections import deque, Counter
import base64

app = Flask(__name__)

class ASLTranscriber:
    def __init__(self, model_path, class_indices_path):
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded!")
        
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        self.idx_to_class = {v: k for k, v in class_indices.items()}
        
        self.prediction_buffer = deque(maxlen=5)
        self.confidence_threshold = 0.7
        
    def preprocess_frame(self, frame):
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    
    def predict(self, frame):
        img = self.preprocess_frame(frame)
        predictions = self.model.predict(img, verbose=0)[0]
        
        confidence = np.max(predictions)
        class_idx = np.argmax(predictions)
        
        if confidence < self.confidence_threshold:
            return None, confidence
        
        # Temporal smoothing
        self.prediction_buffer.append(class_idx)
        
        if len(self.prediction_buffer) >= 3:
            smoothed_prediction = Counter(self.prediction_buffer).most_common(1)[0][0]
            return self.idx_to_class[smoothed_prediction], confidence
        
        return self.idx_to_class[class_idx], confidence

# Initialize the transcriber
transcriber = ASLTranscriber("best_model.keras", "class_indices.json")
transcription = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Predict sign
            sign, confidence = transcriber.predict(frame)
            
            # Add to transcription if confident
            global transcription
            if sign and confidence > 0.85 and (not transcription or transcription[-1] != sign):
                transcription.append(sign)
                print(f"Detected: {sign} ({confidence:.1%})")
            
            # Draw UI on frame
            if sign:
                cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (400, 80), (0, 255, 0), 2)
                cv2.putText(frame, f"Sign: {sign}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Show a sign...", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/get_transcription')
def get_transcription():
    return jsonify({'transcription': ' '.join(transcription)})

@app.route('/clear_transcription')
def clear_transcription():
    global transcription
    transcription = []
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)