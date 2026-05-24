import cv2
import numpy as np
import mediapipe as mp
import pickle
import time

from collections import deque, Counter
from tensorflow.keras.models import load_model

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model("weights/mha_gesturenet.h5")

with open("weights/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ==========================================
# MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

mp_draw = mp.solutions.drawing_utils

# ==========================================
# HEALTHCARE MAP
# ==========================================

GESTURE_INFO = {

    "call": ("Nurse Call", "#ff4b4b"),
    "stop": ("Emergency Stop", "#ff0000"),
    "palm": ("Attention Required", "#ff9800"),
    "fist": ("Pain / Distress", "#ff5252"),
    "like": ("Patient Stable", "#00c853"),
    "dislike": ("Discomfort", "#ff6d00"),
    "mute": ("Cannot Speak", "#d500f9"),
    "peace": ("Need Assistance", "#00b0ff"),
    "ok": ("Confirmation", "#00c853"),
}

# ==========================================
# TEMPORAL SMOOTHING
# ==========================================

SEQUENCE_LENGTH = 10

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

prediction_buffer = deque(maxlen=15)

CONFIDENCE_THRESHOLD = 0.80

# ==========================================
# GENERATE FRAMES
# ==========================================

def generate_frames():

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        final_label = "No Gesture"
        final_conf = 0.0

        # ==========================================
        # HAND DETECTION
        # ==========================================

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(
                        color=(0,255,255),
                        thickness=2,
                        circle_radius=3
                    ),
                    mp_draw.DrawingSpec(
                        color=(255,255,255),
                        thickness=2
                    )
                )

                landmarks = []

                for lm in hand_landmarks.landmark:

                    landmarks.extend([
                        lm.x,
                        lm.y
                    ])

                if len(landmarks) == 42:

                    sequence_buffer.append(landmarks)

                # ==========================================
                # PREDICTION
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

                    prediction = model.predict(
                        input_sequence,
                        verbose=0
                    )

                    predicted_class = np.argmax(prediction)

                    confidence = np.max(prediction)

                    predicted_label = (
                        label_encoder.inverse_transform(
                            [predicted_class]
                        )[0]
                    )

                    # ==========================================
                    # CONFIDENCE FILTER
                    # ==========================================

                    if confidence > CONFIDENCE_THRESHOLD:

                        prediction_buffer.append(
                            predicted_label
                        )

                    # ==========================================
                    # SMOOTHING
                    # ==========================================

                    if len(prediction_buffer) > 5:

                        final_label = Counter(
                            prediction_buffer
                        ).most_common(1)[0][0]

                        final_conf = confidence

        # ==========================================
        # UI COLORS
        # ==========================================

        alert_text = "Monitoring Active"
        alert_color = (0, 255, 0)

        if final_label in GESTURE_INFO:

            alert_text = GESTURE_INFO[final_label][0]

            if final_label in ["stop", "fist"]:
                alert_color = (0, 0, 255)

            elif final_label in ["call", "palm"]:
                alert_color = (0, 165, 255)

            else:
                alert_color = (0, 255, 0)

        # ==========================================
        # PREMIUM OVERLAY
        # ==========================================

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (20, 20),
            (500, 180),
            (20, 20, 20),
            -1
        )

        alpha = 0.55

        frame = cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0
        )

        # ==========================================
        # TEXTS
        # ==========================================

        cv2.putText(
            frame,
            "MHA-GestureNet",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255,255,255),
            3
        )

        cv2.putText(
            frame,
            f"Gesture: {final_label}",
            (40, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {final_conf:.2f}",
            (40, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            alert_text,
            (40, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            alert_color,
            3
        )

        yield frame

    cap.release()