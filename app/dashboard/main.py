import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

st.title("MHA GestureNet")

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

webrtc_streamer(
    key="test",
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

st.success("WebRTC Widget Loaded")
