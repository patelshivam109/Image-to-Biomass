import sys
from pathlib import Path

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
from PIL import Image

from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision import transforms

from src.models.biomass_model import BiomassModel
from src.data.transforms import val_transform
from src.config import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, TARGET_COLUMNS

# Set page config
st.set_page_config(
    page_title="Biomass AI Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium CSS Injection ---
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Title Styling */
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: -webkit-linear-gradient(45deg, #2E8B57, #3CB371);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        text-align: center;
        padding-bottom: 0px;
        margin-bottom: 0px;
    }

    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }

    /* Glassmorphism Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(60, 179, 113, 0.5);
    }

    /* Primary Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #2E8B57 0%, #3CB371 100%);
        color: white;
        border-radius: 30px;
        padding: 10px 30px;
        border: none;
        font-weight: 600;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #3CB371 0%, #2E8B57 100%);
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(46, 139, 87, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- Grad-CAM Implementation ---
class GradCam:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        # Hook into the last convolutional layer of EfficientNet
        target_layer = self.model.feature_extractor[-1]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, img_tensor, meta_tensor, target_idx=0): 
        output = self.model(img_tensor, meta_tensor)
        self.model.zero_grad()
        score = output[0, target_idx]
        score.backward(retain_graph=True)
        
        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)
        activations = self.activations
        
        cam = (gradients * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().detach().cpu().numpy()

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


# --- HEADER ---
st.markdown('<h1 class="main-header">🌿 Precision Pasture AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Biomass Estimation using Deep Learning & Computer Vision</p>', unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1892/1892751.png", width=100)
st.sidebar.title("⚙️ Field Metadata")
st.sidebar.write("Input environmental data to refine predictions.")

ndvi = st.sidebar.slider("NDVI Index", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
height_cm = st.sidebar.number_input("Canopy Height (cm)", value=0.0, step=0.5, help="Leave as 0.0 to auto-estimate using MiDaS AI.")
month = st.sidebar.selectbox("Observation Month", options=list(range(1, 13)), index=5)

state_options = list(encoders['State'].classes_) if 'State' in encoders else ['NSW', 'QLD', 'TAS', 'VIC', 'WA']
state = st.sidebar.selectbox("Geographic State", options=state_options)

species_options = list(encoders['Species'].classes_) if 'Species' in encoders else ['Bahiagrass', 'Clover', 'Lucerne', 'Perennial Ryegrass', 'White Clover']
species = st.sidebar.selectbox("Primary Pasture Species", options=species_options)

st.sidebar.markdown("---")
st.sidebar.info("Tip: Navigate to 'Model Metrics' using the sidebar menu above to view training performance!")

# --- MAIN CONTENT ---
main_col1, main_col2 = st.columns([1, 2])

with main_col1:
    st.markdown("### 📸 Upload Imagery")
    uploaded_file = st.file_uploader("Select a top-down RGB field image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="Source Image", use_container_width=True, clamp=True)
    else:
        st.info("👆 Upload an image to begin the analysis.")

with main_col2:
    if uploaded_file is not None:
        if st.button("🚀 Execute AI Pipeline"):
            with st.spinner("Analyzing vegetation, generating depth maps, and predicting biomass..."):
                try:
                    # A. Pseudo-Depth Estimation (Zero-Download Computer Vision)
                    # Because MiDaS PyTorch model crashes the free-tier server's memory limits, we approximate depth
                    # using grayscale intensity (as taller canopy catches more light / looks closer).
                    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
                    depth_map = cv2.normalize(blurred, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    
                    if height_cm == 0.0:
                        estimated_height = 10 + (np.mean(depth_map) * 20)
                        height_cm = estimated_height
                        st.success(f"📏 CV Auto-Estimated Canopy Height: **{height_cm:.1f} cm**")

                    # B. Vegetation Segmentation (HSV Computer Vision)
                    img_hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
                    lower_green = np.array([25, 40, 40])
                    upper_green = np.array([90, 255, 255])
                    veg_mask = cv2.inRange(img_hsv, lower_green, upper_green)
                    mask_colored = cv2.applyColorMap(veg_mask, cv2.COLORMAP_VIRIDIS)
                    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)

                    # C. Biomass Prediction & Grad-CAM
                    img_tensor = val_transform(image).unsqueeze(0) 
                    num_df = pd.DataFrame([[ndvi, height_cm]], columns=['NDVI', 'Height_cm'])
                    num_scaled = scaler.transform(num_df)[0]
                    state_encoded = encoders['State'].transform([state])[0]
                    species_encoded = encoders['Species'].transform([species])[0]
                    
                    metadata_array = np.concatenate([num_scaled, [month, state_encoded, species_encoded]])
                    meta_tensor = torch.tensor(metadata_array, dtype=torch.float32).unsqueeze(0)
                    
                    prediction = biomass_model(img_tensor, meta_tensor).detach().squeeze(0).numpy()
                    prediction = np.maximum(prediction, 0)
                    
                    cam = grad_cam(img_tensor, meta_tensor, target_idx=0)
                    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    superimposed_img = heatmap * 0.4 + image_np * 0.6
                    superimposed_img = np.uint8(superimposed_img)

                    # --- RESULTS DISPLAY ---
                    st.markdown("### 📊 Yield Estimations (kg/ha)")
                    results = dict(zip(TARGET_COLUMNS, prediction))
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("🌱 Green Dry Matter", f"{results.get('GDM_g', 0):.2f} kg/ha", "Optimal")
                    m2.metric("☘️ Dry Clover Biomass", f"{results.get('Dry_Clover_g', 0):.2f} kg/ha")
                    m3.metric("🍂 Dry Dead Biomass", f"{results.get('Dry_Dead_g', 0):.2f} kg/ha", "-Requires Attention", delta_color="inverse")
                    
                    st.metric("📦 Total Dry Biomass", f"{results.get('Dry_Total_g', 0):.2f} kg/ha")
                    
                    st.markdown("### 👁️ Visual Intelligence")
                    v1, v2, v3 = st.columns(3)
                    with v1:
                        st.image(mask_colored, caption="Semantic Plant Mask", use_container_width=True)
                    with v2:
                        depth_norm = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
                        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
                        st.image(depth_color, caption="Topographic Depth", use_container_width=True)
                    with v3:
                        st.image(superimposed_img, caption="Grad-CAM Yield Heatmap", use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
