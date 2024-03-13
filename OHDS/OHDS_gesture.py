import cv2
import mediapipe as mp

def Print_Finger_Loc(thumb_tip, index_finger_tip, middle_finger_tip, ring_finger_tip, little_finger_tip):
    print("Thumb tip:", thumb_tip)
    print("Index finger tip:", index_finger_tip)
    print("Middle finger tip:", middle_finger_tip)
    print("Ring finger tip:", ring_finger_tip)
    print("Little finger tip:", little_finger_tip)


# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize VideoCapture.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands.
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image.
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Store positions of all finger landmarks into variables.
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            little_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Print positions of finger landmarks.
            Finger_Position_Array = []
            Finger_Position_Array.append

    # Display the image.
    cv2.imshow('Hand Gesture Recognition', image)

    # Press 'q' to exit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
