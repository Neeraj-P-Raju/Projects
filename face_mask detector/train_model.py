import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = []
labels = []


categories = ["face", "unknown_face"]
data_dir = "dataset"


for label, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    for file in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (100, 100))
            data.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue


data = np.array(data) / 255.0
labels = to_categorical(np.array(labels))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2, random_state=42)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=32)


model.save("face_detector_model.h5")
print("✅ Model trained and saved successfully.")