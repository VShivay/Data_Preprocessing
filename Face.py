import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
imp_img = cv2.VideoCapture("elon.jpg")
res, img = imp_img.read()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

# Display the image
cv2.imshow("Elon Musk", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release the video capture object
imp_img.release()
