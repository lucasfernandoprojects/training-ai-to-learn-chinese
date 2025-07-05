import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import serial
import time
import os
import datetime

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Load your trained model
model = tf.keras.models.load_model('YOUR_FINE_TUNED_MODEL_PATH')

# Define class names and their meanings
class_info = {
    'ban_4': {'pinyin': 'bàn', 'meaning': 'half'},
    'cong_2': {'pinyin': 'cóng', 'meaning': 'from'},
    'er_4': {'pinyin': 'èr', 'meaning': 'two'},
    'gan_1': {'pinyin': 'gān', 'meaning': 'dry'},
    'ge_4': {'pinyin': 'gè', 'meaning': 'individual'},
    'ren_2': {'pinyin': 'rén', 'meaning': 'person'},
    'ru_4': {'pinyin': 'rù', 'meaning': 'to enter'},
    'san_1': {'pinyin': 'sān', 'meaning': 'three'},
    'shi_2': {'pinyin': 'shí', 'meaning': 'ten'},
    'yi_1': {'pinyin': 'yī', 'meaning': 'one'}
}

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

cv2.namedWindow("Mandarin Character Classifier", cv2.WINDOW_NORMAL)

# Initialize prediction log
LOG_DIR = "YOUR_LOG_PATH"
LOG_FILE = os.path.join(LOG_DIR, "predictions_log.csv")
CAPTURES_DIR = os.path.join(LOG_DIR, "captures")

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CAPTURES_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w') as f:
        f.write("timestamp,predicted_class,confidence,image_path\n")

# Initialize serial connection
try:
    arduino = serial.Serial('YOUR_ARDUINO_USB_PORT', 9600, timeout=1)
    time.sleep(2)  # Wait for connection to establish
    print("Connected to Arduino")
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    arduino = None

def preprocess_image(img):
    """Preprocess image for model prediction"""
    img = cv2.resize(img, (96, 96))  # Match model's input size
    img = image.img_to_array(img)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)
    return img

def predict_character(img):
    """Make prediction and return top result"""
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return list(class_info.keys())[predicted_class], confidence

def log_prediction(predicted_class, confidence, img):
    """Log prediction details to CSV and save image"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    img_filename = os.path.join(CAPTURES_DIR, f"{predicted_class}_{timestamp}.jpg")
    
    try:
        success = cv2.imwrite(img_filename, img)
        if not success:
            raise IOError(f"Failed to save image: {img_filename}")
        
        with open(LOG_FILE, 'a') as f:
            f.write(f"{timestamp},{predicted_class},{confidence:.4f},{img_filename}\n")
            f.flush()
    except Exception as e:
        print(f"Error saving prediction: {str(e)}")

def send_result_to_arduino(predicted_class):
    """Send the prediction result to Arduino"""
    if arduino is not None:
        result_str = f"RESULT:{predicted_class}\n"  # Just send the class name
        arduino.write(result_str.encode())
        arduino.flush()

def main_loop():
    """Main program loop"""
    print("System ready. Press button on Arduino to capture...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break
            
            # Check for messages from Arduino
            if arduino and arduino.in_waiting > 0:
                message = arduino.readline().decode().strip()
                if message == "CAPTURE":
                    print("Capture command received")
                    
                    # Take and process the photo
                    predicted_class, confidence = predict_character(frame)
                    log_prediction(predicted_class, confidence, frame)
                    
                    # Display result on computer
                    display_frame = frame.copy()
                    text = f"Predicted: {predicted_class} ({confidence:.2%})"
                    cv2.putText(display_frame, text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    meaning = class_info[predicted_class]['meaning']
                    cv2.putText(display_frame, f"Meaning: {meaning}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Send result to Arduino
                    send_result_to_arduino(predicted_class)
                    print(f"Sent result: {predicted_class}")
            
            # Display preview
            display_frame = frame.copy()
            cv2.putText(display_frame, "Press Arduino button to capture", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Mandarin Character Classifier", display_frame)
            
            # Check for ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        if arduino:
            arduino.close()
        print("\nApplication closed successfully.")

if __name__ == "__main__":
    main_loop()