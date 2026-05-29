import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import av

from collections import deque
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MHA · GestureNet",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# THEME TOGGLE
# =========================================================

hdr_left, hdr_right = st.columns([9, 1])

with hdr_right:
    theme = st.toggle("🌙", value=True)

# =========================================================
# THEME SYSTEM
# =========================================================

if theme:
    bg = "#060d14"
    surface = "#0c1825"
    surface2 = "#101f30"
    border = "#1a3048"
    text1 = "#e8f0f8"
    text2 = "#7b9ab8"
    text3 = "#3d5c78"
else:
    bg = "#f4f7fb"
    surface = "#ffffff"
    surface2 = "#edf2f7"
    border = "#d6e0ea"
    text1 = "#0f1720"
    text2 = "#425466"
    text3 = "#7b8794"

# =========================================================
# CSS
# =========================================================

st.markdown(f"""
<style>

html, body, [class*="css"] {{
    background-color: {bg};
    color: {text1};
}}

.stApp {{
    background-color: {bg};
}}

#MainMenu, footer, header {{
    visibility: hidden;
}}

.block-container {{
    padding-top: 1rem;
    max-width: 1400px;
}}

.top-bar {{
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:2rem;
    border-bottom:1px solid {border};
    padding-bottom:1rem;
}}

.logo {{
    font-size:1.6rem;
    font-weight:700;
}}

.subtitle {{
    color:{text2};
    font-size:0.9rem;
}}

.card {{
    background:{surface};
    border:1px solid {border};
    border-radius:18px;
    padding:18px;
}}

.metric-card {{
    background:{surface};
    border:1px solid {border};
    border-radius:14px;
    padding:20px;
    margin-bottom:14px;
}}

.metric-title {{
    font-size:0.7rem;
    color:{text3};
    text-transform:uppercase;
    margin-bottom:10px;
}}

.metric-value {{
    font-size:1.8rem;
    font-weight:700;
}}

.section-label {{
    margin-bottom:10px;
    color:{text3};
    font-size:0.75rem;
    text-transform:uppercase;
}}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_ai_model():

    model = load_model("weights/mha_gesturenet.h5")

    with open("weights/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, label_encoder

model, label_encoder = load_ai_model()

# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# =========================================================
# TEMPORAL BUFFER
# =========================================================

SEQUENCE_LENGTH = 10
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## MHA · GestureNet")

    st.markdown("---")

    st.markdown("### System")

    st.success("AI Online")

    st.markdown("""
    - MediaPipe
    - TensorFlow
    - Browser Webcam
    - Real-time Recognition
    """)

    st.markdown("---")

    st.markdown("### Gestures")

    st.markdown("""
    ✋ CALL  
    🛑 STOP  
    🤐 MUTE
    """)

# =========================================================
# HEADER
# =========================================================

with hdr_left:

    st.markdown(f"""
    <div class="top-bar">
        <div>
            <div class="logo">Healthcare Gesture Alert System</div>
            <div class="subtitle">
                Real-time patient gesture recognition
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# MAIN LAYOUT
# =========================================================

col1, col2 = st.columns([2, 1])

# =========================================================
# RIGHT PANEL PLACEHOLDERS
# =========================================================

with col2:

    st.markdown(
        '<div class="section-label">Detection Panel</div>',
        unsafe_allow_html=True
    )

    gesture_ph = st.empty()
    confidence_ph = st.empty()
    emergency_ph = st.empty()

    gesture_ph.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Detected Gesture</div>
        <div class="metric-value">— —</div>
    </div>
    """, unsafe_allow_html=True)

    confidence_ph.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Confidence</div>
        <div class="metric-value">0%</div>
    </div>
    """, unsafe_allow_html=True)

    emergency_ph.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Emergency Status</div>
        <div class="metric-value">ALL CLEAR</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# LEFT PANEL
# =========================================================

with col1:

    st.markdown(
        '<div class="section-label">Live Camera Feed</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)

    run = st.checkbox(
        "▶ Activate gesture recognition",
        key="run_cam"
    )

    # =====================================================
    # VIDEO PROCESSOR
    # =====================================================

    class GestureProcessor(VideoTransformerBase):

        def transform(self, frame):

            img = frame.to_ndarray(format="bgr24")

            img = cv2.flip(img, 1)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            predicted_label = "No Gesture"
            confidence = 0.0

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:

                    mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    landmarks = []

                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y])

                    if len(landmarks) == 42:
                        sequence_buffer.append(landmarks)

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

                        predicted_label = label_encoder.inverse_transform(
                            [predicted_class]
                        )[0]

                        # =================================
                        # UPDATE RIGHT PANEL
                        # =================================

                        gesture_ph.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">
                                Detected Gesture
                            </div>
                            <div class="metric-value">
                                {predicted_label}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        confidence_ph.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">
                                Confidence
                            </div>
                            <div class="metric-value">
                                {confidence*100:.2f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if predicted_label.upper() in ["STOP", "HELP"]:

                            emergency_ph.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">
                                    Emergency Status
                                </div>
                                <div class="metric-value">
                                    ALERT
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        else:

                            emergency_ph.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">
                                    Emergency Status
                                </div>
                                <div class="metric-value">
                                    ALL CLEAR
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )

    # =====================================================
    # START WEBCAM
    # =====================================================

    if run:

        webrtc_streamer(
            key="gesture-recognition",
            video_transformer_factory=GestureProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_transform=True,
        )

    else:

        st.markdown("""
        <div style="
            height:400px;
            display:flex;
            justify-content:center;
            align-items:center;
            border-radius:14px;
            background:#111827;
            color:#94a3b8;
            font-size:1rem;
        ">
            CAMERA OFFLINE
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
