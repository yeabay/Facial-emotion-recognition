import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import keras
print("Keras version:", keras.__version__)
import cv2
from keras.models import load_model
import numpy as np

# Load our facial emotion recognition model and the emotional mapping to turn into string
emotion_mapping1 = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
model = load_model('aaitproject (1).h5')  #our path
model.compile(optimizer='adam',  # our desired optimizer
              loss=tf.keras.losses.sparse_categorical_crossentropy,  # our desired loss function
              metrics=['accuracy'])
# we create a cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# this is to access the webcam(usually, 0 corresponds to the default built-in webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from the camera or video source.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Preprocess the face region for our model
        face_roi = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
        face_roi = np.expand_dims(np.expand_dims(face_roi, -1), 0) / 255.0

        # Predict emotion using our model
        emotion_probs = model.predict(face_roi)
        emotion_label = np.argmax(emotion_probs)

        # Draw a rectangle around the detected face and display the emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Emotion: {emotion_mapping1[emotion_label]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()