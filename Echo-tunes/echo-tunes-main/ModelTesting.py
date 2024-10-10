import cv2
import numpy as np
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Load the trained model
model = load_model("hand_gesture_model.h5")

# Labels for the gestures
class_labels = ['A', 'B', 'C']  # Add more gesture labels as per your dataset

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Start video capture
while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hand in the frame
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        offset = 20
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        if imgCrop.size > 0:
            imgResize = cv2.resize(imgCrop, (64, 64))
            imgArray = np.array(imgResize) / 255.0  # Normalize
            imgArray = np.expand_dims(imgArray, axis=0)

            # Predict gesture
            predictions = model.predict(imgArray)
            predicted_class = np.argmax(predictions)
            gesture = class_labels[predicted_class]

            # Display the predicted gesture
            cv2.putText(img, f'Gesture: {gesture}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Video Feed", img)

    # Exit the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
