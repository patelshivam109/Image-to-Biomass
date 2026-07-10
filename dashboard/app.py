import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import torch
import pandas as pd
import numpy as np
import cv2
from PIL import Image

from src.models.biomass_model import BiomassModel
from src.visualization.gradcam import GradCam
from src.data.transforms import val_transform
from src.config import TARGET_COLUMNS

# --- Page Config ---
st.set_page_config(
    page_title="Precision Pasture AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Design System CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 30px 0 10px 0;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        color: #E8F5E9;
        margin-bottom: 0;
        letter-spacing: -1px;
    }
    .hero-accent {
        background: linear-gradient(135deg, #66BB6A, #26A69A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #90A4AE;
        font-size: 1rem;
        font-weight: 300;
        margin-top: 5px;
        letter-spacing: 0.5px;
    }

    /* Section Labels */
    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        color: #66BB6A;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 5px;
    }
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #ECEFF1;
        margin-bottom: 15px;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(102, 187, 106, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        border-color: rgba(102, 187, 106, 0.4);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="metric-container"] label {
        color: #90A4AE !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #E8F5E9 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
    }

    /* Button */
    .stButton>button {
        background: linear-gradient(135deg, #2E7D32 0%, #43A047 50%, #66BB6A 100%);
        color: white;
        border-radius: 12px;
        padding: 12px 32px;
        border: none;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(46, 125, 50, 0.5);
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }
    [data-testid="stFileUploader"] section {
        border-radius: 12px;
        border: 2px dashed rgba(102, 187, 106, 0.3);
        padding: 20px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 20, 25, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label {
        color: #B0BEC5 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* Divider */
    .styled-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 187, 106, 0.3), transparent);
        margin: 30px 0;
        border: none;
    }

    /* Status pill */
    .status-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-ready {
        background: rgba(102, 187, 106, 0.15);
        color: #66BB6A;
        border: 1px solid rgba(102, 187, 106, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_assets():
    models_dir = PROJECT_ROOT / "models"
    import joblib
    scaler = joblib.load(models_dir / "scaler.pkl")
    encoders = joblib.load(models_dir / "label_encoders.pkl")
    biomass_model = BiomassModel()
    biomass_model.load_state_dict(torch.load(models_dir / "best_model.pth", map_location=torch.device('cpu'), weights_only=False))
    biomass_model.eval()
    return biomass_model, scaler, encoders

try:
    biomass_model, scaler, encoders = load_assets()
    grad_cam = GradCam(biomass_model)
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# --- HERO ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Precision <span class="hero-accent">Pasture AI</span></div>
    <div class="hero-subtitle">Biomass estimation powered by EfficientNet-B0 and Computer Vision</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<span class="status-pill status-ready">Model Loaded &bull; Ready</span>', unsafe_allow_html=True)
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("""
<div style="padding: 10px 0 20px 0;">
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 700; color: #E8F5E9;">
        Field Parameters
    </div>
    <div style="color: #78909C; font-size: 0.82rem; margin-top: 5px; line-height: 1.5;">
        These environmental inputs provide the AI with essential context about the field conditions, 
        enabling the most accurate biomass prediction possible.
    </div>
</div>
""", unsafe_allow_html=True)

ndvi = st.sidebar.slider("NDVI Index", min_value=0.0, max_value=1.0, value=0.5, step=0.01, 
                          help="Normalized Difference Vegetation Index (0 = bare soil, 1 = dense vegetation)")
height_cm = st.sidebar.number_input("Canopy Height (cm)", value=0.0, step=0.5, 
                                     help="Leave as 0 for automatic estimation from image analysis")
month = st.sidebar.selectbox("Observation Month", options=list(range(1, 13)), index=5)

state_options = list(encoders['State'].classes_) if 'State' in encoders else ['NSW', 'QLD', 'TAS', 'VIC', 'WA']
state = st.sidebar.selectbox("Geographic State", options=state_options)

species_options = list(encoders['Species'].classes_) if 'Species' in encoders else ['Bahiagrass', 'Clover', 'Lucerne', 'Perennial Ryegrass', 'White Clover']
species = st.sidebar.selectbox("Primary Species", options=species_options)

st.sidebar.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="color: #546E7A; font-size: 0.78rem; line-height: 1.6;">
    Use the navigation above to explore <strong>Model Metrics</strong> and the <strong>About</strong> page.
</div>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
upload_col, spacer, results_col = st.columns([4, 0.5, 7])

with upload_col:
    st.markdown('<div class="section-label">Step 1</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Field Image</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Select a top-down RGB field image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="Source Image", use_container_width=True, clamp=True)
        
        st.markdown("")  # spacer
        run_pipeline = st.button("Run Analysis", use_container_width=True)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 40px 20px;">
            <div style="font-size: 2.5rem; margin-bottom: 10px; opacity: 0.3;">&#9653;</div>
            <div style="color: #78909C; font-size: 0.9rem;">
                Upload a top-down RGB image of the pasture field to begin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
        run_pipeline = False

with results_col:
    st.markdown('<div class="section-label">Step 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None and run_pipeline:
        with st.spinner("Processing image through AI pipeline..."):
            try:
                # A. Depth Estimation (OpenCV)
                gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                blurred = cv2.GaussianBlur(gray, (21, 21), 0)
                depth_map = cv2.normalize(blurred, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                
                current_height = height_cm
                if current_height == 0.0:
                    current_height = 10 + (np.mean(depth_map) * 20)
                    st.info(f"Auto-estimated canopy height: **{current_height:.1f} cm**")

                # B. Vegetation Segmentation (HSV)
                img_hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
                lower_green = np.array([25, 40, 40])
                upper_green = np.array([90, 255, 255])
                veg_mask = cv2.inRange(img_hsv, lower_green, upper_green)
                veg_coverage = (np.sum(veg_mask > 0) / veg_mask.size) * 100
                mask_colored = cv2.applyColorMap(veg_mask, cv2.COLORMAP_VIRIDIS)
                mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)

                # C. Biomass Prediction
                img_tensor = val_transform(image).unsqueeze(0) 
                num_df = pd.DataFrame([[ndvi, current_height]], columns=['NDVI', 'Height_cm'])
                num_scaled = scaler.transform(num_df)[0]
                state_encoded = encoders['State'].transform([state])[0]
                species_encoded = encoders['Species'].transform([species])[0]
                
                metadata_array = np.concatenate([num_scaled, [month, state_encoded, species_encoded]])
                meta_tensor = torch.tensor(metadata_array, dtype=torch.float32).unsqueeze(0)
                
                prediction = biomass_model(img_tensor, meta_tensor).detach().squeeze(0).numpy()
                prediction = np.maximum(prediction, 0)
                
                # D. Grad-CAM
                cam = grad_cam(img_tensor, meta_tensor, target_idx=0)
                cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                superimposed_img = np.uint8(heatmap * 0.4 + image_np * 0.6)

                # --- METRICS ---
                results = dict(zip(TARGET_COLUMNS, prediction))
                
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Green Dry Matter", f"{results.get('GDM_g', 0):.1f}", "kg/ha")
                r2.metric("Clover Biomass", f"{results.get('Dry_Clover_g', 0):.1f}", "kg/ha")
                r3.metric("Dead Material", f"{results.get('Dry_Dead_g', 0):.1f}", "kg/ha")
                r4.metric("Total Biomass", f"{results.get('Dry_Total_g', 0):.1f}", "kg/ha")

                st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

                # --- VISUAL OUTPUTS ---
                st.markdown('<div class="section-label">Visual Intelligence</div>', unsafe_allow_html=True)
                v1, v2, v3 = st.columns(3)
                with v1:
                    st.image(mask_colored, caption=f"Vegetation Mask ({veg_coverage:.1f}% coverage)", use_container_width=True)
                with v2:
                    depth_norm = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
                    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
                    st.image(depth_color, caption="Depth Estimation", use_container_width=True)
                with v3:
                    st.image(superimposed_img, caption="Grad-CAM Heatmap", use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        if uploaded_file is None:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 50px 20px;">
                <div style="color: #546E7A; font-size: 0.95rem; line-height: 1.7;">
                    Upload an image and configure field parameters to generate biomass predictions, 
                    vegetation masks, depth maps, and Grad-CAM heatmaps.
                </div>
            </div>
            """, unsafe_allow_html=True)
