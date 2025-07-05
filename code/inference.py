import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import datetime
import os

# Load your trained model
model = tf.keras.models.load_model('YOUR_MODEL_PATH')

# Define class names (replace with your actual classes)
class_names = ['ban_4', 'cong_2', 'er_4', 'gan_1', 'ge_4', 
               'ren_2', 'ru_4', 'san_1', 'shi_2', 'yi_1']

# Initialize webcam
cap = cv2.VideoCapture(0)
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
    return class_names[predicted_class], confidence

def log_prediction(predicted_class, confidence, img):
    """Log prediction details to CSV and save image"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    img_filename = os.path.join(CAPTURES_DIR, f"{predicted_class}_{timestamp}.jpg")
    
    try:
        # First save the image
        success = cv2.imwrite(img_filename, img)
        if not success:
            raise IOError(f"Failed to save image: {img_filename}")
        
        # Then log to CSV - using proper file handling
        with open(LOG_FILE, 'a') as f:
            f.write(f"{timestamp},{predicted_class},{confidence:.4f},{img_filename}\n")
            f.flush()  # Force write to buffer
            os.fsync(f.fileno())  # Now safe because file is still open
            
    except Exception as e:
        print(f"Error saving prediction: {str(e)}")
        # Write to error log with proper file handling
        error_log_path = os.path.join(LOG_DIR, "error_log.txt")
        with open(error_log_path, 'a') as err_f:
            err_f.write(f"{datetime.datetime.now()}: {str(e)}\n")
            err_f.flush()

# State variables
show_result = False
current_prediction = ("", 0)
screenshot = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        # Display appropriate content based on state
        if show_result and screenshot is not None:
            display_frame = screenshot.copy()
            text = f"Predicted: {current_prediction[0]} ({current_prediction[1]:.2%})"
            cv2.putText(display_frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to try again", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, "Press ESC to quit", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            display_frame = frame.copy()
            cv2.putText(display_frame, "Press SPACE to capture", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Mandarin Character Classifier", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Space bar to capture and predict
        if key == ord(' ') and not show_result:
            screenshot = frame.copy()
            current_prediction = predict_character(screenshot)
            log_prediction(current_prediction[0], current_prediction[1], screenshot)
            show_result = True
        
        # Space bar to return to camera
        elif key == ord(' ') and show_result:
            show_result = False
        
        # ESC to exit
        elif key == 27:
            break

finally:
    # Clean up - this will run even if there's an error
    cap.release()
    cv2.destroyAllWindows()
    print("\nApplication closed successfully.")
    print(f"Prediction log saved to: {LOG_FILE}")
    print(f"Captured images saved to: {CAPTURES_DIR}")
    
    # Verification
    log_count = sum(1 for _ in open(LOG_FILE)) - 1  # Subtract header
    img_count = len([f for f in os.listdir(CAPTURES_DIR) if f.endswith('.jpg')])
    print(f"Verification: {log_count} log entries, {img_count} images")
    if log_count != img_count:
        print("Warning: Log and image counts don't match!")