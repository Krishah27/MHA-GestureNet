import streamlit as st
import cv2
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../"
        )
    )
)

from src.inference.streamlit_inference import generate_frames

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="MHA-GestureNet",
    layout="wide"
)

# ==========================================
# THEME
# ==========================================

dark_mode = st.sidebar.toggle(
    "🌙 Dark Mode",
    value=False
)

bg = "#0f172a" if dark_mode else "#f5f7fb"
card = "#111827" if dark_mode else "#ffffff"
text = "#ffffff" if dark_mode else "#111111"
subtext = "#cbd5e1" if dark_mode else "#555555"

st.markdown(f"""
<style>

.stApp {{
    background-color: {bg};
    color: {text};
}}

.main-title {{
    font-size: 52px;
    font-weight: 800;
    color: {text};
}}

.subtitle {{
    color: {subtext};
    font-size: 20px;
}}

.card {{
    background: {card};
    padding: 24px;
    border-radius: 22px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}}

.metric {{
    font-size: 22px;
    font-weight: 700;
    color: {text};
}}

.small {{
    color: {subtext};
}}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class='card'>
<h1 class='main-title'>🏥 MHA-GestureNet</h1>
<p class='subtitle'>
AI Powered Healthcare Gesture Recognition & Emergency Alert System
</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ==========================================
# LAYOUT
# ==========================================

col1, col2 = st.columns([3,1])

with col1:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    run = st.button(
        "▶ Start AI Monitoring"
    )

    FRAME_WINDOW = st.image([])

    if run:

        for frame in generate_frames():

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            FRAME_WINDOW.image(frame)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class='card'>
    <h2 style='color:{text};'>🚨 Active Gestures</h2>

    <p class='metric'>📞 call</p>
    <p class='small'>Nurse Call</p>

    <p class='metric'>✋ stop</p>
    <p class='small'>Emergency Stop</p>

    <p class='metric'>🖐 palm</p>
    <p class='small'>Attention Required</p>

    <p class='metric'>✊ fist</p>
    <p class='small'>Pain Detection</p>

    <p class='metric'>👍 like</p>
    <p class='small'>Patient Stable</p>

    </div>
    """, unsafe_allow_html=True)