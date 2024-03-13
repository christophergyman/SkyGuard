import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def detect_gesture(thumb_tip, index_finger_tip, middle_finger_tip, ring_finger_tip, little_finger_tip):
    # Calculate distances between finger tips.
    thumb_index_distance = ((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5
    index_middle_distance = ((index_finger_tip.x - middle_finger_tip.x) ** 2 + (index_finger_tip.y - middle_finger_tip.y) ** 2) ** 0.5
    middle_ring_distance = ((middle_finger_tip.x - ring_finger_tip.x) ** 2 + (middle_finger_tip.y - ring_finger_tip.y) ** 2) ** 0.5
    ring_little_distance = ((ring_finger_tip.x - little_finger_tip.x) ** 2 + (ring_finger_tip.y - little_finger_tip.y) ** 2) ** 0.5

    # Define thresholds for gesture detection.
    thumbs_up_threshold = 0.1
    thumbs_down_threshold = -0.1
    index_extended_threshold = 0.1
    pinky_extended_threshold = 0.1
    closed_fist_threshold = 0.1

    # Detect gestures.
    thumbs_up = thumb_index_distance < thumbs_up_threshold and index_middle_distance > thumbs_up_threshold and middle_ring_distance > thumbs_up_threshold and ring_little_distance > thumbs_up_threshold
    thumbs_down = thumb_index_distance > thumbs_down_threshold and index_middle_distance < thumbs_down_threshold and middle_ring_distance < thumbs_down_threshold and ring_little_distance < thumbs_down_threshold
    index_extended = index_middle_distance > index_extended_threshold and middle_ring_distance < index_extended_threshold and ring_little_distance < index_extended_threshold
    pinky_extended = index_middle_distance < pinky_extended_threshold and middle_ring_distance > pinky_extended_threshold and ring_little_distance > pinky_extended_threshold
    closed_fist = thumb_index_distance < closed_fist_threshold and index_middle_distance < closed_fist_threshold and middle_ring_distance < closed_fist_threshold and ring_little_distance < closed_fist_threshold

    # Return detected gestures.
    if thumbs_up:
        return "Thumbs Up"
    elif thumbs_down:
        return "Thumbs Down"
    elif index_extended:
        return "Index Finger Extended"
    elif pinky_extended:
        return "Pinky Finger Extended"
    elif closed_fist:
        return "Closed Fist"
    else:
        return None

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

            # Store positions of finger landmarks into variables.
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            little_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Detect gestures.
            gesture = detect_gesture(thumb_tip, index_finger_tip, middle_finger_tip, ring_finger_tip, little_finger_tip)
            if gesture:
                print("Detected Gesture:", gesture)

    # Display the image.
    cv2.imshow('Hand Gesture Recognition', image)

    # Press 'q' to exit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
