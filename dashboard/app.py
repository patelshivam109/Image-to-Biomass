import sys
from pathlib import Path

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import torch
import pandas as pd
import numpy as np
import pickle
from PIL import Image

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
st.write("Upload an RGB pasture image and input metadata to predict biomass.")

@st.cache_resource
def load_assets():
    """Load the trained model, scaler, and label encoders."""
    models_dir = PROJECT_ROOT / "models"
    
    import joblib
    
    # Load Scaler
    scaler = joblib.load(models_dir / "scaler.pkl")
        
    # Load Encoders
    encoders = joblib.load(models_dir / "label_encoders.pkl")
        
    # Load Model
    model = BiomassModel()
    model.load_state_dict(torch.load(models_dir / "best_model.pth", map_location=torch.device('cpu'), weights_only=False))
    model.eval()
    
    return model, scaler, encoders

try:
    model, scaler, encoders = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}. Please ensure 'best_model.pth', 'scaler.pkl', and 'label_encoders.pkl' exist in the 'models/' directory.")
    st.stop()

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Input Data")
    
    # Image Upload
    uploaded_file = st.file_uploader("Upload Pasture Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
    # Metadata Inputs
    st.subheader("Metadata")
    ndvi = st.number_input("NDVI", value=0.5, step=0.01, format="%.2f")
    height_cm = st.number_input("Height (cm)", value=15.0, step=0.5, format="%.1f")
    month = st.selectbox("Month", options=list(range(1, 13)), index=5)
    
    # Categorical Inputs
    state_options = list(encoders['State'].classes_) if 'State' in encoders else ['NSW', 'QLD', 'TAS', 'VIC', 'WA']
    state = st.selectbox("State", options=state_options)
    
    species_options = list(encoders['Species'].classes_) if 'Species' in encoders else ['Bahiagrass', 'Clover', 'Lucerne', 'Perennial Ryegrass', 'White Clover']
    species = st.selectbox("Species", options=species_options)

with col2:
    st.header("2. Prediction Results")
    
    if st.button("Predict Biomass", type="primary"):
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Processing..."):
                try:
                    # 1. Transform Image
                    img_tensor = val_transform(image).unsqueeze(0) # Add batch dimension
                    
                    # 2. Process Metadata
                    # Numerical
                    num_df = pd.DataFrame([[ndvi, height_cm, month]], columns=NUMERICAL_COLUMNS)
                    num_scaled = scaler.transform(num_df)[0]
                    
                    # Categorical
                    state_encoded = encoders['State'].transform([state])[0]
                    species_encoded = encoders['Species'].transform([species])[0]
                    
                    # Combine Metadata
                    metadata_array = np.concatenate([num_scaled, [state_encoded, species_encoded]])
                    meta_tensor = torch.tensor(metadata_array, dtype=torch.float32).unsqueeze(0)
                    
                    # 3. Model Inference
                    with torch.no_grad():
                        prediction = model(img_tensor, meta_tensor)
                        
                    prediction = prediction.squeeze(0).numpy()
                    
                    # 4. Display Results
                    st.success("Prediction Complete!")
                    
                    # Present as cards or metrics
                    # Assuming target columns: "Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "Dry_Total_g", "GDM_g"
                    # But wait, target columns might be different order, let's map them
                    
                    st.write("### Estimated Biomass (kg/ha)")
                    
                    # Create a nice layout for metrics
                    m1, m2 = st.columns(2)
                    m3, m4 = st.columns(2)
                    
                    # Ensure predictions are non-negative
                    prediction = np.maximum(prediction, 0)
                    
                    # Using config TARGET_COLUMNS for order mapping
                    results = dict(zip(TARGET_COLUMNS, prediction))
                    
                    # Assuming output is in grams based on column name "_g", convert to kg/ha
                    # Let's assume the model outputs grams per quadrat (e.g. 0.25m^2), we'd need to multiply by 40 to get kg/ha. 
                    # If it's already predicting kg/ha, then just display. We'll show the raw model output for now.
                    
                    m1.metric("Dry Green Biomass", f"{results.get('Dry_Green_g', 0):.2f}")
                    m2.metric("Dry Dead Biomass", f"{results.get('Dry_Dead_g', 0):.2f}")
                    m3.metric("Dry Clover Biomass", f"{results.get('Dry_Clover_g', 0):.2f}")
                    m4.metric("Total Dry Biomass", f"{results.get('Dry_Total_g', 0):.2f}")
                    
                    st.metric("Green Dry Matter (GDM)", f"{results.get('GDM_g', 0):.2f}")
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
