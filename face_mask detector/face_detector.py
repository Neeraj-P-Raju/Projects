import traceback

try:
    # your existing imports and code...
    
    import cv2
    import numpy as np
    from tensorflow.keras.models import load_model

    model = load_model("face_detector_model.h5")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(frame, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            result = model.predict(face)
            label = "Access Granted" if np.argmax(result) == 0 else "Denied"
            color = (0, 255, 0) if label == "Face" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    with open("error_log.txt", "w") as f:
        f.write(traceback.format_exc())
