import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("MHA GestureNet")

webrtc_streamer(
    key="test",
    media_stream_constraints={"video": True, "audio": False}
)

st.success("WebRTC Widget Loaded")
