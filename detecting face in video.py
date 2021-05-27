
import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture('sample_video.mp4')


while True:
     # Read the frame
    _, img = cap.read()
    
    # Convert to grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect the faces  
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    
    
    # Draw the rectangle around each face  
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0), 4)
    
    # Display  
    cv2.imshow('Video', img)
    
    # Stop if escape key is pressed  
    if cv2.waitKey(1) & 0xff==ord('q'):
        break

cap.release()
