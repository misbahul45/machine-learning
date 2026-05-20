import cv2
import mediapipe as mp
import joblib
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model", "gesture_svm_pipeline.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

model = joblib.load(model_path)

print(f"🔍 Model type: {type(model).__name__}")
print(f"🔍 Classes: {model.classes_ if hasattr(model, 'classes_') else 'N/A'}")

gesture_to_word = {
    "open": "HELLO", "close": "STOP", "point": "YOU", "peace": "GOOD",
    "thumb": "YES", "rock": "COOL",
    "open_inverted": "HELLO", "close_inverted": "STOP", "point_inverted": "YOU",
    "peace_inverted": "GOOD", "thumb_inverted": "YES", "rock_inverted": "COOL"
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

stable_pred = None
counter = 0
current_word = ""
threshold = 12

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_text = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            wrist = raw_coords[0]
            normalized_coords = raw_coords - wrist 
            data = normalized_coords.reshape(1, -1)

            pred = model.predict(data)[0]

            if pred == stable_pred:
                counter += 1
            else:
                stable_pred = pred
                counter = 0

            if counter >= threshold:
                gesture_text = pred
                current_word = gesture_to_word.get(pred, pred)

    cv2.putText(frame, f"Gesture: {gesture_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Word: {current_word}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Stability: {counter}/{threshold}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    cv2.imshow("Gesture AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()