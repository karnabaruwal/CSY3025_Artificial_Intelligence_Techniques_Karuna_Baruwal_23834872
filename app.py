# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt
import io

# ----------------------------
# Config
# ----------------------------
kb_IMG_SIZE = 48
kb_MODEL_PATH = "masked_face_multitask_model_inference.h5"

# Labels
kb_emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
kb_age_labels = ["young", "adult", "older"]
kb_ethnicity_labels = ["asian", "other", "white"]
kb_gender_labels = ["male", "female"]

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def kb_load_trained_model():
    try:
        return load_model(kb_MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None

kb_model = kb_load_trained_model()
if kb_model is None:
    st.stop()

# ----------------------------
# Preprocess
# ----------------------------
def kb_preprocess_image(kb_img: Image.Image):
    kb_img = kb_img.convert("L")  # grayscale
    kb_img = kb_img.resize((kb_IMG_SIZE, kb_IMG_SIZE))
    kb_arr = img_to_array(kb_img) / 255.0
    kb_arr = np.expand_dims(kb_arr, axis=0)
    return kb_arr

# ----------------------------
# Predict
# ----------------------------
def kb_predict(kb_img: Image.Image):
    kb_arr = kb_preprocess_image(kb_img)
    kb_preds = kb_model.predict(kb_arr, verbose=0)

    kb_emo_pred = np.argmax(kb_preds[0], axis=1)[0]
    kb_age_pred = np.argmax(kb_preds[1], axis=1)[0]
    kb_eth_pred = np.argmax(kb_preds[2], axis=1)[0]

    kb_results = {
        "emotion": (kb_emotion_labels[kb_emo_pred], kb_preds[0][0]),
        "age": (kb_age_labels[kb_age_pred], kb_preds[1][0]),
        "ethnicity": (kb_ethnicity_labels[kb_eth_pred], kb_preds[2][0])
    }

    if len(kb_preds) == 4:  # gender included
        kb_gen_score = kb_preds[3][0][0]
        kb_gen_pred = 1 if kb_gen_score >= 0.5 else 0
        kb_results["gender"] = (kb_gender_labels[kb_gen_pred], kb_gen_score)
    return kb_results

# ----------------------------
# Save to History
# ----------------------------
def kb_save_to_history(kb_image: Image.Image, kb_results: dict):
    if "kb_history" not in st.session_state:
        st.session_state.kb_history = []
    kb_buf = io.BytesIO()
    kb_image.save(kb_buf, format="PNG")
    st.session_state.kb_history.append((kb_buf.getvalue(), kb_results))

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Image Classification", layout="wide")
st.markdown("<h1 style='color:#22409a;'>Image Classification</h1>", unsafe_allow_html=True)

# Sidebar navigation
menu = st.sidebar.radio("Navigation", ["Upload Images", "History"])

# ----------------------------
# Custom CSS for Cards
# ----------------------------
st.markdown(
    """
    <style>
    .pred-card {
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: white;
        margin: 5px;
    }
    .emotion { background-color: #1f77b4; }
    .age { background-color: #2ca02c; }
    .ethnicity { background-color: #ff7f0e; }
    .gender { background-color: #d62728; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Upload Page
# ----------------------------
if menu == "Upload Images":
    kb_uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if kb_uploaded_file:
        kb_image = Image.open(kb_uploaded_file)
        st.image(kb_image, caption="Uploaded Image", width=250)

        kb_results = kb_predict(kb_image)
        kb_save_to_history(kb_image, kb_results)

        st.markdown("### 🎯 Predictions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='pred-card emotion'>Emotion: {kb_results['emotion'][0]}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='pred-card age'>Age Group: {kb_results['age'][0]}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='pred-card ethnicity'>Ethnicity: {kb_results['ethnicity'][0]}</div>", unsafe_allow_html=True)
            if "gender" in kb_results:
                st.markdown(f"<div class='pred-card gender'>Gender: {kb_results['gender'][0]} ({kb_results['gender'][1]:.2f})</div>", unsafe_allow_html=True)

        # Confidence Bar Chart (smaller)
        kb_emo_conf = kb_results["emotion"][1]
        kb_fig, kb_ax = plt.subplots(figsize=(5, 3))
        kb_ax.bar(kb_emotion_labels, kb_emo_conf, color="skyblue")
        kb_ax.set_title("Emotion Confidence", fontsize=14)
        kb_ax.set_ylabel("Confidence")
        kb_ax.set_ylim([0, 1])
        for kb_i, kb_v in enumerate(kb_emo_conf):
            kb_ax.text(kb_i, kb_v + 0.01, f"{kb_v:.2f}", ha="center", fontsize=8)
        st.pyplot(kb_fig)

# ----------------------------
# History Page
# ----------------------------
elif menu == "History":
    st.subheader("Detection History")
    if "kb_history" not in st.session_state or len(st.session_state.kb_history) == 0:
        st.info("No history available yet.")
    else:
        if st.button("Clear History"):
            st.session_state.kb_history = []
            st.success("History cleared.")
            st.rerun()

        for kb_idx, (kb_img_bytes, kb_results) in enumerate(st.session_state.kb_history[::-1], 1):
            st.write(f"### Entry {kb_idx}")
            st.image(kb_img_bytes, width=200, caption="Detected Image")

            # Colorful prediction cards in history
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div class='pred-card emotion'>Emotion: {kb_results['emotion'][0]}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='pred-card age'>Age Group: {kb_results['age'][0]}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='pred-card ethnicity'>Ethnicity: {kb_results['ethnicity'][0]}</div>", unsafe_allow_html=True)
                if "gender" in kb_results:
                    st.markdown(f"<div class='pred-card gender'>Gender: {kb_results['gender'][0]} ({kb_results['gender'][1]:.2f})</div>", unsafe_allow_html=True)

            # Download buttons
            kb_pred_text = (
                f"Emotion: {kb_results['emotion'][0]}\n"
                f"Age: {kb_results['age'][0]}\n"
                f"Ethnicity: {kb_results['ethnicity'][0]}\n"
            )
            if "gender" in kb_results:
                kb_pred_text += f"Gender: {kb_results['gender'][0]} ({kb_results['gender'][1]:.2f} probability)\n"

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Image",
                    data=kb_img_bytes,
                    file_name=f"detection_entry_{kb_idx}.png",
                    mime="image/png"
                )
            with col2:
                st.download_button(
                    label="Download Predictions",
                    data=kb_pred_text,
                    file_name=f"detection_entry_{kb_idx}.txt",
                    mime="text/plain"
                )
