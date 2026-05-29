import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

from collections import deque
from tensorflow.keras.models import load_model

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MHA · GestureNet",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# THEME TOGGLE  — must come BEFORE CSS injection
# =========================================================

hdr_left, hdr_right = st.columns([9, 1])

with hdr_right:
    theme = st.toggle("🌙", value=True, help="Dark / Light mode")

# =========================================================
# THEME SYSTEM
# =========================================================

if theme:
    bg       = "#060d14"
    surface  = "#0c1825"
    surface2 = "#101f30"
    border   = "#1a3048"
    text1    = "#e8f0f8"
    text2    = "#7b9ab8"
    text3    = "#3d5c78"
    cam_offline_grad = "linear-gradient(135deg,#060d14 0%,#0c1825 100%)"
else:
    bg       = "#f4f7fb"
    surface  = "#ffffff"
    surface2 = "#edf2f7"
    border   = "#d6e0ea"
    text1    = "#0f1720"
    text2    = "#425466"
    text3    = "#7b8794"
    cam_offline_grad = "linear-gradient(135deg,#edf2f7 0%,#dce8f5 100%)"

# =========================================================
# GLOBAL CSS — Dynamic Theme
# =========================================================

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=Instrument+Sans:wght@400;500;600&display=swap');

/* ── Root tokens ── */
:root {{
    --bg:        {bg};
    --surface:   {surface};
    --surface-2: {surface2};
    --border:    {border};

    --blue:      #1e7fff;
    --blue-dim:  #1e7fff22;
    --teal:      #00c9a7;
    --teal-dim:  #00c9a718;
    --amber:     #ffb830;
    --amber-dim: #ffb83018;
    --red:       #ff4560;
    --red-dim:   #ff456018;

    --text-1:    {text1};
    --text-2:    {text2};
    --text-3:    {text3};

    --mono:      'DM Mono', monospace;
    --display:   'Syne', sans-serif;
    --body:      'Instrument Sans', sans-serif;
    --radius:    12px;
    --radius-lg: 18px;
}}

/* ── Base reset ── */
html, body, [class*="css"] {{
    font-family: var(--body);
    background-color: var(--bg) !important;
    color: var(--text-1);
}}
.stApp {{ background: var(--bg) !important; }}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0.5rem 2.5rem 3rem !important; max-width: 1400px !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: var(--surface); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 99px; }}

/* ════════════════════════════════════════
   THEME TOGGLE ROW
════════════════════════════════════════ */
/* Push the toggle column to top-right and align with header */
div[data-testid="column"]:last-child {{
    display: flex;
    justify-content: flex-end;
    align-items: center;
    padding-top: 0.2rem;
}}
/* Style the toggle label */
[data-testid="stToggle"] label {{
    font-family: var(--mono) !important;
    font-size: 1rem !important;
    color: var(--text-2) !important;
    cursor: pointer;
}}

/* ════════════════════════════════════════
   TOP HEADER BAR
════════════════════════════════════════ */
.top-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 1.6rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.8rem;
}}
.top-bar-left {{
    display: flex;
    align-items: center;
    gap: 16px;
}}
.logo-mark {{
    width: 44px; height: 44px;
    background: var(--blue);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--display);
    font-weight: 800;
    font-size: 18px;
    color: #fff;
    letter-spacing: -1px;
    box-shadow: 0 0 20px #1e7fff44;
    flex-shrink: 0;
}}
.header-title {{
    font-family: var(--display);
    font-weight: 700;
    font-size: 1.35rem;
    color: var(--text-1);
    letter-spacing: -0.5px;
    line-height: 1.1;
}}
.header-sub {{
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--text-3);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 2px;
}}
.header-badges {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: flex-end;
}}
.badge {{
    padding: 5px 12px;
    border-radius: 99px;
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    border: 1px solid;
    display: flex; align-items: center; gap: 6px;
    white-space: nowrap;
}}
.badge-green {{ color: var(--teal);  border-color: var(--teal);  background: var(--teal-dim);  }}
.badge-blue  {{ color: var(--blue);  border-color: var(--blue);  background: var(--blue-dim);  }}
.dot {{ width:6px; height:6px; border-radius:50%; background:currentColor; animation: pulse 2s infinite; }}
@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.4}} }}

/* ════════════════════════════════════════
   SECTION LABELS
════════════════════════════════════════ */
.section-label {{
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px;
}}
.section-label::after {{
    content:''; flex:1; height:1px; background: var(--border);
}}

/* ════════════════════════════════════════
   CAMERA FEED CARD
════════════════════════════════════════ */
.cam-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
}}
.cam-top-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    border-bottom: 1px solid var(--border);
    background: var(--surface-2);
}}
.cam-dot-group {{ display:flex; gap:7px; }}
.cam-dot {{ width:10px; height:10px; border-radius:50%; }}
.cam-inner {{ padding: 16px; }}
.cam-offline {{
    aspect-ratio: 16/9;
    display: flex; flex-direction:column;
    align-items: center; justify-content: center;
    gap: 12px;
    background: {cam_offline_grad};
    border-radius: 8px;
    color: var(--text-3);
    font-family: var(--mono);
    font-size: 0.8rem;
    letter-spacing: 0.06em;
}}
.cam-offline-icon {{ font-size: 2.5rem; opacity: .3; }}

/* ════════════════════════════════════════
   CHECKBOX / ACTIVATE BUTTON
════════════════════════════════════════ */
.stCheckbox {{ margin-top: 14px !important; }}
.stCheckbox label {{
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-2) !important;
    letter-spacing: 0.05em !important;
}}

/* ════════════════════════════════════════
   METRIC CARDS
════════════════════════════════════════ */
.metric-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color .25s, box-shadow .25s;
}}
.metric-card::before {{
    content:'';
    position: absolute;
    top:0; left:0; right:0; height:2px;
}}
.metric-card.blue::before  {{ background: var(--blue);  box-shadow: 0 0 12px var(--blue);  }}
.metric-card.teal::before  {{ background: var(--teal);  box-shadow: 0 0 12px var(--teal);  }}
.metric-card.amber::before {{ background: var(--amber); box-shadow: 0 0 12px var(--amber); }}
.metric-card.red::before   {{ background: var(--red);   box-shadow: 0 0 12px var(--red);   }}

.metric-label {{
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 8px;
}}
.metric-value {{
    font-family: var(--display);
    font-weight: 700;
    font-size: 1.6rem;
    line-height: 1;
    letter-spacing: -1px;
    color: var(--text-1);
}}
.metric-value.blue  {{ color: var(--blue);  }}
.metric-value.teal  {{ color: var(--teal);  }}
.metric-value.amber {{ color: var(--amber); }}
.metric-value.red   {{ color: var(--red);   }}
.metric-sub {{
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--text-3);
    margin-top: 6px;
}}

/* ════════════════════════════════════════
   EMERGENCY ALERT CARD
════════════════════════════════════════ */
.alert-card {{
    border-radius: var(--radius);
    padding: 16px 20px;
    display: flex; align-items: center; gap: 14px;
    border: 1px solid;
    margin-top: 4px;
}}
.alert-card.normal  {{ background: var(--teal-dim);  border-color: var(--teal);  }}
.alert-card.warning {{
    background: var(--amber-dim);
    border-color: var(--amber);
    animation: blink-amber .8s ease-in-out infinite;
}}
.alert-card.critical {{
    background: var(--red-dim);
    border-color: var(--red);
    animation: blink-red .6s ease-in-out infinite;
}}
@keyframes blink-amber {{ 0%,100%{{box-shadow:0 0 0 0 transparent}} 50%{{box-shadow:0 0 14px 2px #ffb83055}} }}
@keyframes blink-red   {{ 0%,100%{{box-shadow:0 0 0 0 transparent}} 50%{{box-shadow:0 0 18px 3px #ff456055}} }}

.alert-icon {{ font-size: 1.6rem; }}
.alert-title {{
    font-family: var(--display);
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: -0.3px;
}}
.alert-card.normal  .alert-title {{ color: var(--teal);  }}
.alert-card.warning .alert-title {{ color: var(--amber); }}
.alert-card.critical .alert-title {{ color: var(--red);  }}
.alert-desc {{
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.05em;
    color: var(--text-2);
    margin-top: 3px;
}}

/* ════════════════════════════════════════
   HISTORY PANEL
════════════════════════════════════════ */
.history-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    max-height: 320px;
    overflow-y: auto;
}}
.history-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 10px;
    border-radius: 7px;
    margin-bottom: 4px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    transition: border-color .2s;
}}
.history-row:hover {{ border-color: var(--blue); }}
.history-gesture {{
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--text-1);
    letter-spacing: 0.04em;
}}
.history-tag {{
    font-family: var(--mono);
    font-size: 0.6rem;
    padding: 2px 8px;
    border-radius: 99px;
    border: 1px solid var(--border);
    color: var(--text-3);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.history-tag.alert {{ border-color: var(--red);   color: var(--red);   background: var(--red-dim);   }}
.history-tag.warn  {{ border-color: var(--amber); color: var(--amber); background: var(--amber-dim); }}

/* ════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════ */
[data-testid="stSidebar"] {{
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}}
[data-testid="stSidebar"] * {{ color: var(--text-2) !important; }}

.sidebar-brand {{
    padding: 1.2rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}}
.sidebar-title {{
    font-family: var(--display) !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    color: var(--text-1) !important;
    letter-spacing: -0.5px;
    line-height: 1.2;
}}
.sidebar-version {{
    font-family: var(--mono) !important;
    font-size: 0.62rem !important;
    color: var(--text-3) !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 3px;
}}
.status-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 9px 0;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.03em;
}}
.status-key      {{ color: var(--text-3); }}
.status-val-ok   {{ color: var(--teal);  font-weight: 500; }}
.status-val-blue {{ color: var(--blue);  font-weight: 500; }}
.status-val-text {{ color: var(--text-1); }}

.sidebar-section {{ margin-top: 1.4rem; }}
.sidebar-sec-title {{
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 10px;
}}
.gesture-ref-item {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
}}
.gesture-ref-icon {{ font-size: 1.3rem; }}
.gesture-ref-name {{
    font-family: var(--display);
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-1) !important;
    letter-spacing: -0.3px;
}}
.gesture-ref-desc {{
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--text-3) !important;
    letter-spacing: 0.03em;
    margin-top: 1px;
}}

/* ════════════════════════════════════════
   Streamlit widget overrides
════════════════════════════════════════ */
[data-testid="stImage"] img {{
    border-radius: 8px;
    width: 100% !important;
}}
[data-testid="stMetricValue"] {{
    font-family: var(--display) !important;
    font-weight: 700 !important;
    color: var(--blue) !important;
}}
[data-testid="stMetricLabel"] {{
    font-family: var(--mono) !important;
    font-size: 0.65rem !important;
    color: var(--text-3) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}}
[data-testid="stAlert"] {{
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}}

/* separator */
hr {{ border-color: var(--border) !important; margin: 1.8rem 0 !important; }}

/* Subheader override */
h2, h3 {{
    font-family: var(--display) !important;
    letter-spacing: -0.5px !important;
    color: var(--text-1) !important;
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
# MEDIAPIPE SETUP
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
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-title">MHA · GestureNet</div>
        <div class="sidebar-version">v1.1 · Clinical Build</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div>
        <div class="status-row">
            <span class="status-key">SYSTEM</span>
            <span class="status-val-ok">● ONLINE</span>
        </div>
        <div class="status-row">
            <span class="status-key">MODEL</span>
            <span class="status-val-blue">MHA-GestureNet</span>
        </div>
        <div class="status-row">
            <span class="status-key">SEQUENCE</span>
            <span class="status-val-text">10 frames</span>
        </div>
        <div class="status-row">
            <span class="status-key">BACKEND</span>
            <span class="status-val-ok">MediaPipe + TF</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-sec-title">Gesture Reference</div>
        <div class="gesture-ref-item">
            <div class="gesture-ref-icon">✋</div>
            <div>
                <div class="gesture-ref-name">CALL</div>
                <div class="gesture-ref-desc">Triggers nurse call alert</div>
            </div>
        </div>
        <div class="gesture-ref-item">
            <div class="gesture-ref-icon">🛑</div>
            <div>
                <div class="gesture-ref-name">STOP</div>
                <div class="gesture-ref-desc">Emergency stop signal</div>
            </div>
        </div>
        <div class="gesture-ref-item">
            <div class="gesture-ref-icon">🤐</div>
            <div>
                <div class="gesture-ref-name">MUTE</div>
                <div class="gesture-ref-desc">Patient cannot speak</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-sec-title">Notes</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Ensure adequate lighting. Position hand clearly in frame. System processes 10-frame sequences for reliable prediction.")

# =========================================================
# TOP HEADER  (inside left column of hdr_left / hdr_right)
# =========================================================

with hdr_left:
    st.markdown("""
    <div class="top-bar">
        <div class="top-bar-left">
            <div class="logo-mark">MH</div>
            <div>
                <div class="header-title">Healthcare Gesture Alert System</div>
                <div class="header-sub">Real-time patient gesture recognition · ICU / Ward monitoring</div>
            </div>
        </div>
        <div class="header-badges">
            <div class="badge badge-green"><span class="dot"></span>AI Active</div>
            <div class="badge badge-blue">MHA-GestureNet v1.1</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# MAIN LAYOUT
# =========================================================

col1, col2 = st.columns([2, 1], gap="large")

# =========================================================
# LEFT COLUMN — CAMERA
# =========================================================

with col1:
    st.markdown('<div class="section-label">Live Camera Feed</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="cam-card">
        <div class="cam-top-bar">
            <div class="cam-dot-group">
                <div class="cam-dot" style="background:#ff4560"></div>
                <div class="cam-dot" style="background:#ffb830"></div>
                <div class="cam-dot" style="background:#00c9a7"></div>
            </div>
            <span style="font-family:var(--mono);font-size:0.65rem;color:var(--text-3);letter-spacing:.08em">PATIENT CAM · LIVE</span>
            <span style="font-family:var(--mono);font-size:0.65rem;color:var(--text-3)">CAM-01</span>
        </div>
        <div class="cam-inner">
    """, unsafe_allow_html=True)

    run = st.checkbox("▶  Activate gesture recognition", key="run_cam")

    FRAME_WINDOW = st.image([])

    if not run:
        st.markdown("""
        <div class="cam-offline">
            <div class="cam-offline-icon">📷</div>
            <span>CAMERA OFFLINE — toggle above to start</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

# =========================================================
# RIGHT COLUMN — METRICS & STATUS
# =========================================================

with col2:
    st.markdown('<div class="section-label">Detection Panel</div>', unsafe_allow_html=True)

    gesture_ph    = st.empty()
    confidence_ph = st.empty()
    emergency_ph  = st.empty()
    fps_ph        = st.empty()

# =========================================================
# INITIAL STATE — right panel placeholders
# =========================================================

with gesture_ph.container():
    st.markdown("""
    <div class="metric-card blue">
        <div class="metric-label">Detected Gesture</div>
        <div class="metric-value blue">——</div>
        <div class="metric-sub">Waiting for input</div>
    </div>
    """, unsafe_allow_html=True)

with confidence_ph.container():
    st.markdown("""
    <div class="metric-card teal" style="margin-top:12px">
        <div class="metric-label">Confidence Score</div>
        <div class="metric-value teal">0.00%</div>
        <div class="metric-sub">Model certainty</div>
    </div>
    """, unsafe_allow_html=True)

with emergency_ph.container():
    st.markdown("""
    <div class="section-label" style="margin-top:20px">Emergency Status</div>
    <div class="alert-card normal">
        <div class="alert-icon">✅</div>
        <div class="alert-text-wrap">
            <div class="alert-title">ALL CLEAR</div>
            <div class="alert-desc">No alerts detected</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with fps_ph.container():
    st.markdown("""
    <div class="metric-card blue" style="margin-top:12px">
        <div class="metric-label">Processing Speed</div>
        <div class="metric-value blue">— FPS</div>
        <div class="metric-sub">Frames per second</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PREDICTION HISTORY STATE
# =========================================================

prediction_history = []

# =========================================================
# WEBCAM LOOP
# =========================================================

if run:
    camera = cv2.VideoCapture(0)
    prev_time = 0

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("⚠ Failed to access webcam. Check connection.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        predicted_label = "No Gesture"
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                if len(landmarks) == 42:
                    sequence_buffer.append(landmarks)

                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    input_sequence = np.array(sequence_buffer, dtype=np.float32)
                    input_sequence = np.expand_dims(input_sequence, axis=0)
                    prediction = model.predict(input_sequence, verbose=0)
                    predicted_class = np.argmax(prediction)
                    confidence = np.max(prediction)
                    predicted_label = label_encoder.inverse_transform([predicted_class])[0]


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
