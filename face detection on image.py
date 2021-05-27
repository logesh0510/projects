

import cv2

imagePath = "/home/pi/Pictures/test.jpeg"
cascPath = "/home/pi/projects/haarcascade_frontalface_default.xml"


faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=1,
    minSize=(15, 15)
)


print ("Found {} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 5)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
