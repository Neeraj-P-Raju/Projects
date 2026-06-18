import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or your trained model
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorbike, bus, truck

cap = cv2.VideoCapture(0)
vehicle_count = 0
counted_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            track_id = int(box.id[0].item()) if box.id is not None else None

            if cls not in VEHICLE_CLASSES or track_id is None:
                continue

            # Bounding box and center
            x1, y1, x2, y2 = box.xyxy[0]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw box and center
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Count vehicle once when first detected
            if track_id not in counted_ids:
                vehicle_count += 1
                counted_ids.add(track_id)
                print(f"Vehicle #{vehicle_count} counted!")

    cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.imshow("Vehicle Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
