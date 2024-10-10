import cv2
import os
from cvzone.HandTrackingModule import HandDetector  # Import the Hand Detector
import tensorflow as tf
from tensorflow.keras import layers, models

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize the hand detector with cvzone
detector = HandDetector(maxHands=1)  # Detect only one hand

# Folder for saving images
Data = "Data/C"  # Folder where images will be saved
gesture_name = "C"  # Change this for each gesture you're capturing (e.g., A, B, C, etc.)
img_count = 0  # To count how many images you've saved

# Create the folder if it doesn't exist
if not os.path.exists(Data):
    os.makedirs(Data)

# Start video capture
while True:
    success, img = cap.read()  # Capture the video frame
    if not success:
        break

    # Detect hand in the frame
    hands, img = detector.findHands(img)  # Detect hands and draw landmarks

    if hands:
        # Get the first hand detected
        hand = hands[0]
        # Get the bounding box of the hand (x, y, width, height)
        x, y, w, h = hand['bbox']

        # Crop the image to the hand region with some offset
        offset = 20  # Offset to include some area around the hand
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        # Ensure the crop is valid and within bounds
        if imgCrop.size > 0:
            # Display the cropped hand
            cv2.imshow("Cropped Hand", imgCrop)

            # Save the cropped hand image when spacebar is pressed
            if cv2.waitKey(1) & 0xFF == ord(' '):  # Press spacebar to save
                img_path = os.path.join(Data, f"{gesture_name}_{img_count}.jpg")  # Save with a unique name
                cv2.imwrite(img_path, imgCrop)  # Save the image to the folder
                img_count += 1  # Increase the image counter
                print(f"Saved {img_path}")  # Print confirmation message

    # Display the full frame with hand landmarks
    cv2.imshow("Video Feed", img)

    # Exit the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

