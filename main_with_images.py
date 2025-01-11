from ultralytics import YOLO
import cv2

def detect_guns(image_path):
    
    model = YOLO('best.pt')  
    
    image = cv2.imread(image_path)
    results = model(image, conf=0.65)[0]
    
    for box in results.boxes:
        confidence = box.conf.item()
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        conf_text = f'Gun: {confidence:.2f}'
        cv2.putText(image, conf_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Gun Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = 'ww.jpg'  
    detect_guns(image_path)