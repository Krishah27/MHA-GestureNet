import streamlit as st

# Page Config
st.set_page_config(
    page_title="Healthcare Gesture Alert System",
    layout="wide"
)

# Title
st.title("🏥 Healthcare Gesture Alert System")

st.markdown("---")

# Sidebar
st.sidebar.header("System Status")
st.sidebar.success("AI System Active")

# Layout
col1, col2 = st.columns([2, 1])

# Left Column
with col1:
    st.subheader("📷 Live Webcam Feed")

    # Webcam placeholder
    st.info("Webcam feed will appear here")

# Right Column
with col2:

    st.subheader("🖐 Detected Gesture")
    st.success("No Gesture Detected")

    st.subheader("🎯 Confidence Score")
    st.metric(label="Confidence", value="0%")

    st.subheader("⚠ Emergency Status")
    st.error("No Emergency")

    st.subheader("⚡ FPS")
    st.metric(label="FPS", value="0")

# Bottom Section
st.markdown("---")

st.subheader("📜 Prediction History")

st.write("No predictions yet.")