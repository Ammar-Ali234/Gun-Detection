from ultralytics import YOLO
import cv2
import time

def detect_guns_from_webcam():
    model = YOLO('C:\\Users\\HP\\OneDrive\\Desktop\\Portfolio\\Gun\\best.pt')  
    gun_class_id = 0  
    
    cap = cv2.VideoCapture(0)  # Access the webcam
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        start_time = time.time()
        results = model(frame, conf=0.4)[0]
        
        for box in results.boxes:
            class_id = int(box.cls.item())  
            if class_id == gun_class_id:  
                confidence = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                conf_text = f'Gun: {confidence:.2f}'
                cv2.putText(frame, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        fps = 1 / (time.time() - start_time)
        fps_text = f'FPS: {fps:.2f}'
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Gun Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_guns_from_webcam()
