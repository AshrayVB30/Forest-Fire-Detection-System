import streamlit as st
import requests
from PIL import Image
import io
import base64

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="🔥 Forest Fire Detection", layout="centered")

st.title("🔥 Forest Fire Detection System")
st.write("Upload an image and the system will predict whether there is a fire or not, with a heatmap overlay.")

uploaded_file = st.file_uploader("Upload a forest image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Fire"):
        with st.spinner("Analyzing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    fire_prob = result["class_probabilities"]["Fire"]
                    nofire_prob = result["class_probabilities"]["No Fire"]

                    st.subheader("📊 Prediction Result")
                    if prediction == "Fire":
                        st.error(f"🔥 Fire Detected! (Confidence: {confidence:.2%})")
                    else:
                        st.success(f"✅ No Fire Detected (Confidence: {confidence:.2%})")

                    if fire_prob > 0.3:
                        st.warning(f"⚠️ Warning: Fire Probability = {fire_prob:.2%}")

                    # Class probabilities
                    st.write("### Class Probabilities")
                    st.write(f"🔥 Fire: **{fire_prob:.2%}**")
                    st.progress(min(max(fire_prob, 0.0), 1.0))
                    st.write(f"🌲 No Fire: **{nofire_prob:.2%}**")
                    st.progress(min(max(nofire_prob, 0.0), 1.0))

                    # ✅ Show Grad-CAM heatmap
                    heatmap_base64 = result.get("heatmap", None)
                    if heatmap_base64:
                        st.write("### 🔥 Fire Detection Heatmap")
                        heatmap_bytes = base64.b64decode(heatmap_base64)
                        heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
                        st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
                    else:
                        st.warning("⚠️ No heatmap received from backend.")

                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"⚠️ Could not connect to backend: {e}")
