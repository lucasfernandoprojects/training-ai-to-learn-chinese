import cv2
import os

def capture_images(output_dir, base_name):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print(f"Press SPACE to capture. Press ESC to quit.")
    
    count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Display preview
        cv2.imshow("Webcam Preview (SPACE=capture, ESC=quit)", frame)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break
        elif key == 32:  # SPACE to capture
            img_name = f"{base_name}_{count:03d}.jpg"
            img_path = os.path.join(output_dir, img_name)
            cv2.imwrite(img_path, frame)
            print(f"Saved: {img_path}")
            count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_dir = input("Enter directory to save images (e.g., ./dataset): ").strip()
    base_name = input("Enter base name for images (e.g., 'zhong'): ").strip()
    capture_images(output_dir, base_name)