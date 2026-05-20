import cv2
import mediapipe as mp

# ==========================================
# MEDIAPIPE SETUP
# ==========================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ==========================================
# WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # ==========================================
    # HAND DETECTION
    # ==========================================

    results = hands.process(rgb)

    # ==========================================
    # DRAW LANDMARKS
    # ==========================================

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

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

            print(
                "Landmark Vector Length:",
                len(landmarks)
            )

    # ==========================================
    # SHOW WINDOW
    # ==========================================

    cv2.imshow(
        "MediaPipe Hands",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# RELEASE
# ==========================================

cap.release()

cv2.destroyAllWindows()