from ultralytics import YOLO
import cv2

def detect_guns(image_path):
    # Load your custom trained model
    model = YOLO('best.pt')  # Replace with your model path
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Perform detection with confidence threshold of 0.5
    results = model(image, conf=0.65)[0]
    
    # Process and draw detections
    for box in results.boxes:
        # Get confidence
        confidence = box.conf.item()
        
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Add confidence text
        conf_text = f'Gun: {confidence:.2f}'
        cv2.putText(image, conf_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Show image
    cv2.imshow('Gun Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Use the function
if __name__ == "__main__":
    image_path = 'ww.jpg'  # Replace with your image path
    detect_guns(image_path)