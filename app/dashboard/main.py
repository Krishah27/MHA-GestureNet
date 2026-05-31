import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

st.title("MHA GestureNet")

model = load_model("weights/mha_gesturenet.h5")

st.success("Model Loaded Successfully 🚀")
