# app.py

import streamlit as st
from PIL import Image
import torch
from utils import load_model, predict_image
from ui import render_ui

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load():
    return load_model("./models/model.pth", device)

model = load()

# Render UI and handle prediction
uploaded_file = render_ui()

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        prediction = predict_image(model, image, device)

    st.success(f"🔍 Predicted Class: **{prediction}**")
