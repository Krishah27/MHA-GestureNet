import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import av
from collections import deque
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

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

html, body, [class*="css"] {{
    font-family: var(--body);
    background-color: var(--bg) !important;
    color: var(--text-1);
}}
.stApp {{ background: var(--bg) !important; }}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0.5rem 2.5rem 3rem !important; max-width: 1400px !important; }}

::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: var(--surface); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 99px; }}

div[data-testid="column"]:last-child {{
    display: flex;
    justify-content: flex-end;
    align-items: center;
    padding-top: 0.2rem;
}}
[data-testid="stToggle"] label {{
    font-family: var(--mono) !important;
    font-size: 1rem !important;
    color: var(--text-2) !important;
    cursor: pointer;
}}

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

/* WebRTC video styling */
[data-testid="stVideo"] video,
.stWebRtcStreamer video {{
    border-radius: 8px;
    width: 100% !important;
}}
div[data-testid="stWebRtcStreamer"] > div {{
    border-radius: 8px;
    overflow: hidden;
}}

hr {{ border-color: var(--border) !important; margin: 1.8rem 0 !important; }}
h2, h3 {{
    font-family: var(--display) !important;
    letter-spacing: -0.5px !important;
    color: var(--text-1) !important;
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE — shared results from video callback
# =========================================================

if "predicted_label" not in st.session_state:
    st.session_state.predicted_label = "No Gesture"
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "fps" not in st.session_state:
    st.session_state.fps = 0.0

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
mp_draw  = mp.solutions.drawing_utils

# =========================================================
# VIDEO PROCESSOR — runs in WebRTC thread
# =========================================================

import time
from streamlit_webrtc import VideoProcessorBase

class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.sequence_buffer = deque(maxlen=10)
        self.predicted_label = "No Gesture"
        self.confidence      = 0.0
        self.prev_time       = time.time()
        self.fps             = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # FPS
        now = time.time()
        self.fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now

        rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        self.predicted_label = "No Gesture"
        self.confidence      = 0.0

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_lm, mp_hands.HAND_CONNECTIONS)
                lms = []
                for lm in hand_lm.landmark:
                    lms.extend([lm.x, lm.y])
                if len(lms) == 42:
                    self.sequence_buffer.append(lms)
                if len(self.sequence_buffer) == 10:
                    seq   = np.expand_dims(np.array(self.sequence_buffer, dtype=np.float32), 0)
                    pred  = model.predict(seq, verbose=0)
                    cls   = np.argmax(pred)
                    self.confidence      = float(np.max(pred))
                    self.predicted_label = label_encoder.inverse_transform([cls])[0]

        # Overlay on frame
        label_text = f"{self.predicted_label}  {self.confidence*100:.1f}%"
        cv2.putText(img, label_text, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 200, 167), 2, cv2.LINE_AA)

        # Push results to session state via a simple global (WebRTC runs in a thread)
        st.session_state.predicted_label = self.predicted_label
        st.session_state.confidence      = self.confidence
        st.session_state.fps             = self.fps

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =========================================================
# RTC CONFIGURATION  (STUN for NAT traversal on Render)
# =========================================================

RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

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
# TOP HEADER
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

# ─── LEFT: Camera ────────────────────────────────────────

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

    # ── Browser-based webcam (works on Render / any cloud) ──
    ctx = webrtc_streamer(
        key="gesture-cam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=GestureProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.markdown("</div></div>", unsafe_allow_html=True)

# ─── RIGHT: Detection panel ───────────────────────────────

with col2:
    st.markdown('<div class="section-label">Detection Panel</div>', unsafe_allow_html=True)

    # Pull latest results from session state
    label      = st.session_state.predicted_label
    confidence = st.session_state.confidence
    fps_val    = st.session_state.fps

    # ── Gesture card ──
    st.markdown(f"""
    <div class="metric-card blue">
        <div class="metric-label">Detected Gesture</div>
        <div class="metric-value blue">{label if label != "No Gesture" else "——"}</div>
        <div class="metric-sub">{"Live detection" if label != "No Gesture" else "Waiting for input"}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence card ──
    st.markdown(f"""
    <div class="metric-card teal" style="margin-top:12px">
        <div class="metric-label">Confidence Score</div>
        <div class="metric-value teal">{confidence*100:.2f}%</div>
        <div class="metric-sub">Model certainty</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Emergency status ──
    EMERGENCY_GESTURES = {"STOP", "CALL", "MUTE"}
    WARNING_GESTURES   = {"PAIN", "HELP"}

    if label in EMERGENCY_GESTURES:
        alert_cls   = "critical"
        alert_icon  = "🚨"
        alert_title = "EMERGENCY ALERT"
        alert_desc  = f"Patient signalled: {label}"
    elif label in WARNING_GESTURES:
        alert_cls   = "warning"
        alert_icon  = "⚠️"
        alert_title = "ATTENTION NEEDED"
        alert_desc  = f"Patient gesture: {label}"
    else:
        alert_cls   = "normal"
        alert_icon  = "✅"
        alert_title = "ALL CLEAR"
        alert_desc  = "No alerts detected"

    st.markdown(f"""
    <div class="section-label" style="margin-top:20px">Emergency Status</div>
    <div class="alert-card {alert_cls}">
        <div class="alert-icon">{alert_icon}</div>
        <div class="alert-text-wrap">
            <div class="alert-title">{alert_title}</div>
            <div class="alert-desc">{alert_desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── FPS card ──
    fps_display = f"{fps_val:.1f} FPS" if fps_val > 0 else "— FPS"
    st.markdown(f"""
    <div class="metric-card blue" style="margin-top:12px">
        <div class="metric-label">Processing Speed</div>
        <div class="metric-value blue">{fps_display}</div>
        <div class="metric-sub">Frames per second</div>
    </div>
    """, unsafe_allow_html=True)
