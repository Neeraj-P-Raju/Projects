import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("mask_detector_model.h5")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


labels = ["Mask", "No Mask"]
colors = [(0, 255, 0), (0, 0, 255)]


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100)) 
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        result = model.predict(face_input)
        label = np.argmax(result)
        confidence = result[0][label]

        cv2.rectangle(frame, (x, y), (x+w, y+h), colors[label], 2)
        cv2.putText(frame, f"{labels[label]} ({confidence*100:.2f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[label], 2)

    cv2.imshow("Face Mask Detector", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
