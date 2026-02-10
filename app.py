
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet import preprocess_input

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5", compile=False)

model = load_model()

with open("class_names.txt") as f:
    class_names = f.read().splitlines()

st.title("🧠 Brain Tumor Detection System")
st.write("Upload a Brain MRI image to detect tumor type")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    img = image.resize((224,224))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    idx = np.argmax(preds)
    conf = np.max(preds) * 100

    st.subheader("Prediction")
    st.write(f"Tumor Type: **{class_names[idx].upper()}**")
    st.write(f"Confidence: **{conf:.2f}%**")
