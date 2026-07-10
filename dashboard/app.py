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
    page_title="Biomass Prediction Dashboard",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Image-to-Biomass Prediction")
st.write("Upload an RGB pasture image to generate segmentation masks, depth maps, and predict biomass.")

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
        
    def __call__(self, img_tensor, meta_tensor, target_idx=0): # Default target is Dry Green (0)
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
    """Load all models (Biomass, Segmentation, MiDaS), scaler, and label encoders."""
    models_dir = PROJECT_ROOT / "models"
    import joblib
    
    # 1. Biomass Model Assets
    scaler = joblib.load(models_dir / "scaler.pkl")
    encoders = joblib.load(models_dir / "label_encoders.pkl")
    biomass_model = BiomassModel()
    biomass_model.load_state_dict(torch.load(models_dir / "best_model.pth", map_location=torch.device('cpu'), weights_only=False))
    biomass_model.eval()
    
    # 2. DeepLabV3+ Segmentation Model (Pre-trained)
    seg_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    seg_model.eval()
    
    # 3. MiDaS Depth Estimation Model
    # Using 'MiDaS_small' for fast CPU inference
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    
    return biomass_model, scaler, encoders, seg_model, midas, midas_transforms

try:
    biomass_model, scaler, encoders, seg_model, midas, midas_transforms = load_assets()
    grad_cam = GradCam(biomass_model)
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Input Data")
    
    uploaded_file = st.file_uploader("Upload Pasture Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="Original RGB Image", use_container_width=True)
        
    st.subheader("Metadata")
    ndvi = st.number_input("NDVI", value=0.5, step=0.01, format="%.2f")
    height_cm = st.number_input("Height (cm) - Optional (will be estimated if 0)", value=0.0, step=0.5, format="%.1f")
    month = st.selectbox("Month", options=list(range(1, 13)), index=5)
    
    state_options = list(encoders['State'].classes_) if 'State' in encoders else ['NSW', 'QLD', 'TAS', 'VIC', 'WA']
    state = st.selectbox("State", options=state_options)
    
    species_options = list(encoders['Species'].classes_) if 'Species' in encoders else ['Bahiagrass', 'Clover', 'Lucerne', 'Perennial Ryegrass', 'White Clover']
    species = st.selectbox("Species", options=species_options)

with col2:
    st.header("2. Analysis & Prediction")
    
    if st.button("Run Full Pipeline", type="primary"):
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Running AI Pipeline (Segmentation, Depth, Biomass)..."):
                try:
                    # ==========================================
                    # A. MiDaS Depth Estimation
                    # ==========================================
                    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    input_batch = midas_transforms(img_cv)
                    with torch.no_grad():
                        depth_prediction = midas(input_batch)
                        depth_prediction = torch.nn.functional.interpolate(
                            depth_prediction.unsqueeze(1),
                            size=img_cv.shape[:2],
                            mode="bicubic",
                            align_corners=False,
                        ).squeeze()
                    depth_map = depth_prediction.cpu().numpy()
                    
                    # Estimate height from depth if user left it as 0
                    if height_cm == 0.0:
                        # Normalize depth to a pseudo-height range (10cm - 30cm) for demonstration
                        estimated_height = 10 + (np.mean(depth_map) / np.max(depth_map)) * 20
                        height_cm = estimated_height
                        st.info(f"📏 Auto-Estimated Canopy Height from MiDaS: **{height_cm:.1f} cm**")

                    # ==========================================
                    # B. DeepLabV3 Vegetation Mask
                    # ==========================================
                    # Standard torchvision transform for DeepLab
                    seg_preprocess = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    input_tensor = seg_preprocess(image).unsqueeze(0)
                    with torch.no_grad():
                        seg_output = seg_model(input_tensor)['out'][0]
                    seg_predictions = seg_output.argmax(0).byte().cpu().numpy()
                    
                    # Create a generic vegetation mask (merging all "plant" like classes if possible, 
                    # but for visualization we'll just colorize the DeepLab output map)
                    mask_colored = cv2.applyColorMap(seg_predictions * 10, cv2.COLORMAP_VIRIDIS)
                    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)

                    # ==========================================
                    # C. Biomass Prediction & Grad-CAM
                    # ==========================================
                    # 1. Transform Image
                    img_tensor = val_transform(image).unsqueeze(0) 
                    
                    # 2. Process Metadata
                    num_df = pd.DataFrame([[ndvi, height_cm]], columns=['NDVI', 'Height_cm'])
                    num_scaled = scaler.transform(num_df)[0]
                    state_encoded = encoders['State'].transform([state])[0]
                    species_encoded = encoders['Species'].transform([species])[0]
                    
                    metadata_array = np.concatenate([num_scaled, [month, state_encoded, species_encoded]])
                    meta_tensor = torch.tensor(metadata_array, dtype=torch.float32).unsqueeze(0)
                    
                    # 3. Model Inference & Grad-CAM
                    prediction = biomass_model(img_tensor, meta_tensor).detach().squeeze(0).numpy()
                    prediction = np.maximum(prediction, 0)
                    
                    # Generate Grad-CAM for "Dry Green Biomass" (Index 0)
                    cam = grad_cam(img_tensor, meta_tensor, target_idx=0)
                    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    superimposed_img = heatmap * 0.4 + image_np * 0.6
                    superimposed_img = np.uint8(superimposed_img)

                    # ==========================================
                    # D. Display Visual Results
                    # ==========================================
                    st.write("### 🖼️ Visual Analysis")
                    v1, v2, v3 = st.columns(3)
                    
                    with v1:
                        st.image(mask_colored, caption="DeepLabV3+ Semantic Segmentation", use_container_width=True)
                    with v2:
                        # Normalize depth map for display
                        depth_norm = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
                        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
                        st.image(depth_color, caption="MiDaS Depth Map", use_container_width=True)
                    with v3:
                        st.image(superimposed_img, caption="Grad-CAM Biomass Heatmap", use_container_width=True)

                    # ==========================================
                    # E. Display Numerical Results
                    # ==========================================
                    st.write("### 📊 Estimated Biomass (kg/ha)")
                    m1, m2 = st.columns(2)
                    m3, m4 = st.columns(2)
                    
                    results = dict(zip(TARGET_COLUMNS, prediction))
                    
                    m1.metric("Dry Green Biomass", f"{results.get('Dry_Green_g', 0):.2f}")
                    m2.metric("Dry Dead Biomass", f"{results.get('Dry_Dead_g', 0):.2f}")
                    m3.metric("Dry Clover Biomass", f"{results.get('Dry_Clover_g', 0):.2f}")
                    m4.metric("Total Dry Biomass", f"{results.get('Dry_Total_g', 0):.2f}")
                    
                    st.metric("Green Dry Matter (GDM)", f"{results.get('GDM_g', 0):.2f}")
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
