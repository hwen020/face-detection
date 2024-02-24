import cv2
# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the classifier was loaded successfully
if face_cascade.empty():
    raise IOError("Unable to load the face cascade classifier xml file")

# Read the image
img = cv2.imread('test.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces on the grayscale image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the image with face rectangles
cv2.imshow('img', img)
cv2.waitKey(0)

# Save the image with face rectangles
cv2.imwrite("face_detected.jpg", img)
