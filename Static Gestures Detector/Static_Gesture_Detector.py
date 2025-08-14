import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
import torch 
import torch.nn as nn
import json

with open("StaticGestures/gesture_index_relation.json", "r") as f:
    gesture_index_relation = json.load(f)

class StaticGesturesDetector(nn.Module):
    def __init__(self, input_size, num_classes):
        super(StaticGesturesDetector, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, X):
        X = self.relu(self.fc1(X))
        X = self.relu(self.fc2(X))
        X = self.fc3(X)
        return X
    
def normalize_hand_keypoints(data_point):
    # Convert list to numpy array
    data_point = np.array(data_point)

    # Split into x and y coordinates
    x_coords = data_point[:21]
    y_coords = data_point[21:]

    # Stack into (21, 2) array
    keypoints = np.stack([x_coords, y_coords], axis=1)  # shape: (21, 2)

    # Centralize using wrist (landmark 0)
    origin = keypoints[0]
    centralized = keypoints - origin

    # Normalize scale by max distance from wrist
    scale = np.linalg.norm(centralized, axis=1).max()
    if scale == 0:
        scale = 1
    normalized = centralized / scale

    # Flatten back to original format: [x0..x20, y0..y20]
    x_norm = normalized[:, 0]
    y_norm = normalized[:, 1]
    return np.concatenate([x_norm, y_norm]).tolist()
    
input_size = 42
num_classes = 5
SGD = StaticGesturesDetector(input_size, num_classes)
SGD.load_state_dict(torch.load("StaticGestures/static_gesture_detector_model.pth"))
SGD.eval()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                    min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y])

                data = np.array(data)
                data = data.flatten()
                data = data.tolist()
                data = normalize_hand_keypoints(data)

                if len(data) == input_size: 
                    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad(): 
                        output = SGD(data)
                        predicted_class = torch.argmax(output, dim=1).item()
                    
                    predicted_gesture = gesture_index_relation[str(predicted_class)]

                    cv2.putText(frame, f"Gesture: {predicted_gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        
        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
    