import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="About the Model", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: -webkit-linear-gradient(45deg, #FF6347, #FF4500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 20px;
    }
    .content-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">About the Project</h1>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("What Problem Does This Solve?")
    st.write("""
    Traditionally, estimating pasture biomass requires a labor-intensive process known as 'clipping and weighing'. 
    A farmer must physically cut a square meter of grass, dry it in an oven to remove moisture, and weigh it on a scale to determine the available feed in kilograms per hectare (kg/ha).
    
    This application automates that entire process using **Computer Vision and Deep Learning**. By simply uploading a top-down RGB image of the pasture and providing some basic environmental metadata, the AI instantly predicts the dry biomass yield. This saves immense amounts of time and money while allowing for much larger scale monitoring.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("The AI Architecture")
    st.write("""
    The core intelligence of this application relies on a **Dual-Input Neural Network** built with PyTorch.
    
    1. **Visual Feature Extraction:** The model uses an `EfficientNet-B0` backbone to process the RGB image. EfficientNet is a state-of-the-art convolutional neural network that is highly optimized to extract complex visual patterns (like leaf texture, canopy density, and grass coverage) with extreme efficiency.
    2. **Metadata Processing Branch:** Images alone can be deceiving (e.g., grass in winter vs. summer). We built a separate neural network branch consisting of fully connected linear layers to process environmental metadata (NDVI, Canopy Height, State, and Species).
    3. **Feature Fusion & Regression:** The visual features extracted by EfficientNet are flattened and mathematically concatenated (fused) with the processed metadata. This fused intelligence is then passed through a final regression head to predict the exact continuous numerical values (in kg/ha) of the various biomass categories.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
with st.container():
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("Visual Intelligence Tools")
    st.write("""
    To make the AI's decisions transparent, this dashboard employs two advanced visual tools:
    
    * **HSV Semantic Segmentation:** We use a traditional computer vision Color-Space algorithm to dynamically isolate green vegetation from the soil background, providing an instant visual mask of the living plant matter.
    * **Grad-CAM (Gradient-weighted Class Activation Mapping):** This algorithm hooks into the final convolutional layers of the EfficientNet model during prediction. It calculates the gradients to generate a heatmap over the original image, showing exactly *where* the AI was looking when it made its biomass prediction.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.info("Built for Precision Agriculture. Powered by PyTorch & Streamlit.")
