import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)

previous_landmarks = None

motion_threshold = 0.005
dynamic_frames = 3 
static_frames = 15 
count = 0
motion_count = 0
status = "Static Gesture"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        current_landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

        if previous_landmarks is not None:
            movement = np.linalg.norm(current_landmarks - previous_landmarks, axis=1).mean()

            if movement > motion_threshold:
                motion_count += 1
                print(f"motion_count : {motion_count} {status}")
                count = 0  
            else:
                count += 1
                print(f"static_count : {count} {status}")
                motion_count = 0 
            
            if motion_count > dynamic_frames:
                status = "Dynamic Gesture"
            
            if count > static_frames:
                status = "Static Gesture"

            cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        previous_landmarks = current_landmarks

    else:
        cv2.putText(frame, "not detected", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()