import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from src.explainability import generate_gradcam, save_and_display_gradcam, generate_clinical_report
from tensorflow.keras.models import load_model

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Medical AI Diagnostic Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #2c3e50;
        background-color: #161b22;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2869/2869815.png", width=100)
    st.title("🏥 Diagnostics")
    st.info("AI-Powered Assistant for Chest X-Ray Analysis.")
    
    selected_page = st.radio("Navigation", ["Home & Predict", "Performance Metrics", "About the AI"])
    
    st.divider()
    st.markdown("### Model Status")
    if os.path.exists('models/best_medical_model.keras'):
        st.success("✅ Model: MobileNetV2-Prepped")
    else:
        st.warning("⚠️ Model: Not Found (Train first)")

# --- MAIN CONTENT ---
if selected_page == "Home & Predict":
    st.title("🩺 AI Medical Diagnostic Engine")
    st.write("Upload a Chest X-ray scan for instant clinical analysis.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Scan")
        uploaded_file = st.file_uploader("Choose a JPG/PNG file...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Scan", use_column_width=True)
            
            if st.button("🚀 Run AI Diagnosis"):
                if not os.path.exists('models/best_medical_model.keras'):
                    st.error("Model file not found. Please run training first.")
                else:
                    with st.spinner('Analyzing scan...'):
                        # Load Model
                        model = load_model('models/best_medical_model.keras')
                        
                        # Preprocess
                        img_path = "temp_prediction.jpg"
                        image.save(img_path)
                        img = cv2.imread(img_path)
                        img_resized = cv2.resize(img, (224, 224))
                        img_normalized = img_resized / 255.0
                        img_batch = np.expand_dims(img_normalized, axis=0)
                        
                        # Predict
                        prediction = model.predict(img_batch)[0][0]
                        diagnosis = "PNEUMONIA DETECTED" if prediction > 0.5 else "NORMAL (HEALTHY)"
                        confidence = prediction if prediction > 0.5 else (1 - prediction)
                        
                        st.session_state['diagnosis'] = diagnosis
                        st.session_state['confidence'] = confidence
                        st.session_state['prediction_done'] = True
                        
                        # Grad-CAM
                        heatmap = generate_gradcam(model, img_batch, 'Conv_1')
                        save_and_display_gradcam(img_path, heatmap, cam_path="temp_gradcam.png")
                        
                        # Generate Report
                        generate_clinical_report(img_path, diagnosis, confidence, heatmap, report_path="temp_report.png")

    with col2:
        st.subheader("📊 AI Findings")
        if 'prediction_done' in st.session_state:
            diag_color = "red" if "PNEUMONIA" in st.session_state['diagnosis'] else "green"
            st.markdown(f"""
                <div class="prediction-box">
                    <h2 style='color:{diag_color};'>{st.session_state['diagnosis']}</h2>
                    <h3>Confidence: {st.session_state['confidence']*100:.2f}%</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            st.image("temp_gradcam.png", caption="Diagnostic Heatmap (Grad-CAM)")
            
            with open("temp_report.png", "rb") as file:
                st.download_button(
                    label="📥 Download Clinical Report",
                    data=file,
                    file_name="diagnosis_report.png",
                    mime="image/png"
                )
        else:
            st.info("Awaiting scan upload and diagnostic trigger...")

elif selected_page == "Performance Metrics":
    st.title("📈 Model Performance Dashboard")
    
    if os.path.exists('outputs/training_history.png'):
        tabs = st.tabs(["Learning Curves", "Confusion Matrix", "ROC-AUC", "Metric Comparison"])
        
        with tabs[0]:
            st.image("outputs/training_history.png", use_column_width=True)
        with tabs[1]:
            st.image("outputs/confusion_matrix.png", use_column_width=True)
        with tabs[2]:
            st.image("outputs/roc_curve.png", use_column_width=True)
        with tabs[3]:
            st.image("outputs/metric_comparison.png", use_column_width=True)
    else:
        st.error("No metrics found. Please train the model to generate performance data.")

elif selected_page == "About the AI":
    st.title("ℹ️ About the Diagnostic System")
    st.markdown("""
    ### Architecture
    The system utilizes **Transfer Learning** on the **MobileNetV2** architecture. 
    It has been fine-tuned on clinical datasets to recognize specific bilateral patterns of lung infections.

    ### Explainability (XAI)
    We use **Grad-CAM** (Gradient-weighted Class Activation Mapping) to produce heatmaps. 
    This allows clinicians to verify that the AI is looking at valid anatomical indicators rather than noise or artifacts.

    ### Technical Specs
    - **Backbone**: MobileNetV2 (ImageNet weights)
    - **Input Size**: 224x224x3
    - **Optimization**: Adam (LR=1e-4)
    - **Libraries**: TensorFlow, OpenCV, Streamlit
    """)
