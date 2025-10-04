import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import requests # New import

# --- MODEL DOWNLOADING LOGIC ---
def download_model(url, save_path):
    # Display a message while downloading
    with st.spinner(f"Downloading model from {url}... This may take a moment."):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.sidebar.success("Model downloaded successfully!")
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {e}")
            st.stop()

st.set_page_config(layout="wide")
st_autorefresh(interval=1000, key="clockfreshener")
st.title("🚀 Space Station Safety Object Detector")

# --- MODEL PATH AND URL ---
# Define where the model will be saved locally in the Streamlit environment
MODEL_PATH = "best.pt"
# PASTE THE DIRECT DOWNLOAD URL YOU COPIED FROM HUGGING FACE HERE
MODEL_URL = "https://huggingface.co/pookyboy72/spacestation-ai-app/resolve/main/best.pt" 

# Check if the model file exists. If not, download it.
if not os.path.exists(MODEL_PATH):
    download_model(MODEL_URL, MODEL_PATH)

# Load the model from the local path
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading the model. Details: {e}")
    st.stop()

# --- SIDEBAR AND UI (No changes needed below this line) ---
with st.sidebar:
    st.header("Dashboard")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%A, %B %d, %Y")
    st.metric("Time (IST)", current_time)
    st.metric("Date", current_date)
    st.success("Model loaded successfully!")

st.markdown("---")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
# ... The rest of your app code is the same ...
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    if st.button('Detect Objects'):
        with st.spinner('Processing...'):
            results = model(image)
            annotated_image = results[0].plot()
            annotated_image_rgb = annotated_image[..., ::-1]
            with col2:
                st.image(annotated_image_rgb, caption='Detected Objects', use_container_width=True)
