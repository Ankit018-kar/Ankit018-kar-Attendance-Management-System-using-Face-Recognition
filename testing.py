import cv2
import numpy as np

# Use the correct method for creating the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained model
recognizer.read('TrainingImageLabel/trainner.yml')

# Set up the face cascade for detecting faces
cascadePath = r'C:\Users\ankit\OneDrive\Documents\Desktop\VS CODE\image processing_face\haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

# Set up the font for displaying the ID
font = cv2.FONT_HERSHEY_SIMPLEX

# Start the webcam feed
cam = cv2.VideoCapture(0)

while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    
    for (x, y, w, h) in faces:
        # Predict the ID of the detected face
        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

        # Draw a rectangle around the face and display the predicted ID
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 7)
        cv2.putText(im, str(Id), (x, y - 40), font, 2, (255, 255, 255), 3)

    # Display the frame with the face detection and recognition
    cv2.imshow('im', im)

    # Press 'q' to exit the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cam.release()
cv2.destroyAllWindows()
