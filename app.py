import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import requests
import pytz

# --- CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Safety Detector",
    page_icon="🚀",
    layout="wide"
)

# --- HELPER FUNCTIONS ---
def download_model(url, save_path):
    with st.spinner(f"Downloading model... This may take a moment."):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.sidebar.success("Model downloaded successfully!")
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {e}")
            st.stop()

# --- LOGIN LOGIC ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def display_login_form():
    st.title("Welcome to the Safety Detector App")
    st.markdown("Please enter username and password to proceed.")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")
        if submitted and username and password:
            st.session_state['logged_in'] = True
            st.rerun()
        elif submitted:
            st.error("Please enter both a username and password.")

# --- MAIN APP FLOW ---
if not st.session_state['logged_in']:
    display_login_form()
else:
    # --- MAIN APPLICATION UI ---
    st_autorefresh(interval=1000, key="clockfreshener")

    st.title("🚀 Space Station Safety Object Detector")
    st.markdown("This application uses a trained YOLO model to detect vital safety equipment.")

    with st.sidebar:
        st.header("Dashboard")
        # Timezone-aware clock
        utc_now = datetime.now(pytz.utc)
        ist_now = utc_now.astimezone(pytz.timezone("Asia/Kolkata"))
        current_time = ist_now.strftime("%H:%M:%S")
        current_date = ist_now.strftime("%A, %B %d, %Y")
        st.metric("Time (IST)", current_time)
        st.metric("Date", current_date)
        
        if st.button("Log Out"):
            st.session_state['logged_in'] = False
            st.rerun()

    # --- MODEL PATH AND URL ---
    MODEL_PATH = "best.pt"
    # PASTE THE DIRECT DOWNLOAD URL YOU COPIED FROM HUGGING FACE HERE
    MODEL_URL = "https://huggingface.co/pookyboy72/spacestation-ai-app/resolve/main/best.pt" 

    if not os.path.exists(MODEL_PATH):
        download_model(MODEL_URL, MODEL_PATH)

    try:
        model = YOLO(MODEL_PATH)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    st.markdown("---")
    input_source = st.radio("Select Input Source", ("Upload Image", "Live Webcam Photo"), horizontal=True)
    img_file_buffer = None

    if input_source == "Upload Image":
        img_file_buffer = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    elif input_source == "Live Webcam Photo":
        img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        col1, col2 = st.columns(2)
        image = Image.open(img_file_buffer)
        
        with col1:
            st.image(image, caption='Input Image', use_container_width=True)

        if st.button('Detect Objects'):
            with st.spinner('Processing...'):
                results = model(image)
                annotated_image = results[0].plot()
                annotated_image_rgb = annotated_image[..., ::-1]
                
                with col2:
                    st.image(annotated_image_rgb, caption='Detected Objects', use_container_width=True)
