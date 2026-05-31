import streamlit as st
from tensorflow.keras.models import load_model

st.title("MHA GestureNet")

try:
    model = load_model("weights/mha_gesturenet.h5")
    st.success("Model Loaded Successfully 🚀")
except Exception as e:
    st.error(f"Model Load Error: {e}")
