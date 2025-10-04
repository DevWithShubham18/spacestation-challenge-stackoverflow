import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import glob
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Safety Detector",
    page_icon="🚀",
    layout="wide"
)

# --- HELPER FUNCTION ---
def find_latest_run():
    run_folders = glob.glob(os.path.join('runs', 'detect', 'train*'))
    if not run_folders:
        return None
    latest_run_folder = max(run_folders, key=lambda f: int(''.join(filter(str.isdigit, f)) or 0))
    model_path = os.path.join(latest_run_folder, 'weights', 'best.pt')
    return model_path

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
    st.markdown("This application uses a trained YOLO model to detect vital safety equipment and automatically loads the latest trained model.")

    with st.sidebar:
        st.header("Dashboard")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_date = now.strftime("%A, %B %d, %Y")
        st.metric("Last Update Time (IST)", current_time)
        st.metric("Date", current_date)
        
        if st.button("Log Out"):
            st.session_state['logged_in'] = False
            st.rerun()

    model_path = find_latest_run()

    if model_path and os.path.exists(model_path):
        st.sidebar.success(f"Loaded model from: {os.path.basename(os.path.dirname(os.path.dirname(model_path)))}")
        try:
            model = YOLO(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
    else:
        st.error("No trained model ('best.pt') found. Please complete a training run.")
        st.stop()

    st.markdown("---")

    input_source = st.radio(
        "Select Input Source",
        ("Upload Image", "Live Webcam Photo"),
        horizontal=True
    )

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