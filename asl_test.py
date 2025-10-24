# asl_test.py
import cv2
import numpy as np
import tensorflow as tf
import json
from collections import deque, Counter

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
    
    def run_webcam(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        print("\n" + "="*60)
        print("🎥 ASL REAL-TIME TRANSCRIBER")
        print("="*60)
        print("Press 'q' to quit")
        print("Starting webcam...\n")
        
        transcription = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict
            sign, confidence = self.predict(frame)
            
            # Display
            display_frame = frame.copy()
            
            if sign:
                # Draw semi-transparent background for text
                cv2.rectangle(display_frame, (10, 10), (400, 80), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (10, 10), (400, 80), (0, 255, 0), 2)
                
                # Display prediction
                text = f"Sign: {sign}"
                conf_text = f"Confidence: {confidence:.1%}"
                
                cv2.putText(display_frame, text, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, conf_text, (20, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add to transcription
                if confidence > 0.85 and (not transcription or transcription[-1] != sign):
                    transcription.append(sign)
                    print(f"Detected: {sign} ({confidence:.1%})")
            else:
                cv2.putText(display_frame, "Show a sign...", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display transcription at bottom
            if transcription:
                trans_text = ' '.join(transcription[-10:])  # Last 10 signs
                cv2.putText(display_frame, f"Text: {trans_text}", (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('ASL Transcriber', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("Final Transcription:")
        print(' '.join(transcription))
        print("="*60)

# Run it!
if __name__ == "__main__":
    # Update these paths to where you downloaded the files
    MODEL_PATH = "best_model.keras"
    CLASS_INDICES_PATH = "class_indices.json"
    
    transcriber = ASLTranscriber(MODEL_PATH, CLASS_INDICES_PATH)
    transcriber.run_webcam()