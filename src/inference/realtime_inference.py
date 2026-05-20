import cv2
import numpy as np
import mediapipe as mp
import pickle

from collections import deque
from tensorflow.keras.models import load_model

# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading MHA-GestureNet Model...")

model = load_model(
    "weights/mha_gesturenet.h5"
)

# ==========================================
# LOAD LABEL ENCODER
# ==========================================

with open(
    "weights/label_encoder.pkl",
    "rb"
) as f:

    label_encoder = pickle.load(f)

print("\nModel Loaded Successfully.")

# ==========================================
# MEDIAPIPE SETUP
# ==========================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ==========================================
# TEMPORAL BUFFER
# ==========================================

SEQUENCE_LENGTH = 10

sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)

# ==========================================
# WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

print("\n===================================")
print("REAL-TIME HEALTHCARE AI STARTED")
print("===================================")

while True:

    success, frame = cap.read()

    if not success:
        break

    # ==========================================
    # FLIP FRAME
    # ==========================================

    frame = cv2.flip(frame, 1)

    # ==========================================
    # RGB CONVERSION
    # ==========================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # ==========================================
    # HAND DETECTION
    # ==========================================

    results = hands.process(rgb)

    predicted_label = "No Gesture"

    confidence = 0.0

    # ==========================================
    # HAND FOUND
    # ==========================================

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # ==========================================
            # DRAW LANDMARKS
            # ==========================================

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ==========================================
            # EXTRACT LANDMARKS
            # ==========================================

            landmarks = []

            for lm in hand_landmarks.landmark:

                landmarks.extend([
                    lm.x,
                    lm.y
                ])

            # ==========================================
            # VALIDATE FEATURE SIZE
            # ==========================================

            if len(landmarks) == 42:

                sequence_buffer.append(
                    landmarks
                )

            # ==========================================
            # PREDICT WHEN BUFFER FULL
            # ==========================================

            if len(sequence_buffer) == SEQUENCE_LENGTH:

                input_sequence = np.array(
                    sequence_buffer,
                    dtype=np.float32
                )

                input_sequence = np.expand_dims(
                    input_sequence,
                    axis=0
                )

                # ==========================================
                # MODEL PREDICTION
                # ==========================================

                prediction = model.predict(
                    input_sequence,
                    verbose=0
                )

                predicted_class = np.argmax(
                    prediction
                )

                confidence = np.max(
                    prediction
                )

                predicted_label = (
                    label_encoder.inverse_transform(
                        [predicted_class]
                    )[0]
                )

    # ==========================================
    # DISPLAY PREDICTION
    # ==========================================

    cv2.putText(
        frame,
        f"Gesture: {predicted_label}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # ==========================================
    # HEALTHCARE ALERTS
    # ==========================================

    if predicted_label == "call":

        cv2.putText(
            frame,
            "NURSE CALL DETECTED",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    elif predicted_label == "stop":

        cv2.putText(
            frame,
            "EMERGENCY STOP",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    elif predicted_label == "mute":

        cv2.putText(
            frame,
            "PATIENT CANNOT SPEAK",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # ==========================================
    # SHOW WINDOW
    # ==========================================

    cv2.imshow(
        "MHA-GestureNet Healthcare AI",
        frame
    )

    # ==========================================
    # QUIT
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# RELEASE
# ==========================================

cap.release()

cv2.destroyAllWindows()