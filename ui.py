# ui.py

import streamlit as st

def render_ui():
    st.title("🗑️ Garbage Classifier")
    st.subheader("Upload an image to classify it into one of 10 garbage types.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    return uploaded_file
