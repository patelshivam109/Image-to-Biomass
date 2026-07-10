import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import json
from PIL import Image

st.set_page_config(page_title="Model Metrics", page_icon="📈", layout="wide")

st.markdown("""
<style>
    /* Premium Styling inherited from main app */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: -webkit-linear-gradient(45deg, #1E90FF, #00BFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📈 Model Training Analytics</h1>', unsafe_allow_html=True)
st.write("Review the regression model's learning curves and validation performance.")

# --- LOAD TRAINING HISTORY ---
history_path = PROJECT_ROOT / "models" / "training_history.json"
if history_path.exists():
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    df = pd.DataFrame({
        'Epoch': epochs,
        'Training Loss': history['train_loss'],
        'Validation Loss': history['val_loss']
    }).set_index('Epoch')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Learning Curve")
        st.line_chart(df, height=350, use_container_width=True)
        
    with col2:
        st.subheader("Final Metrics")
        final_train = history['train_loss'][-1]
        final_val = history['val_loss'][-1]
        best_val = min(history['val_loss'])
        best_epoch = history['val_loss'].index(best_val) + 1
        
        st.metric("Final Training Loss", f"{final_train:.2f}")
        st.metric("Final Validation Loss", f"{final_val:.2f}")
        st.metric("Best Validation Loss", f"{best_val:.2f}", f"Achieved at Epoch {best_epoch}", delta_color="normal")
else:
    st.warning("Training history not found. Train the model to see learning curves.")

st.markdown("---")
st.markdown("### 🖼️ Evaluation Plots")
st.write("Visualizations generated during the model testing phase.")

plots_dir = PROJECT_ROOT / "outputs" / "plots"

# We will display some of the key plots
key_plots = [
    ("Actual vs Predicted", "actual_vs_predicted.png"),
    ("Residual Analysis", "residual_plot.png"),
    ("Feature Correlation Heatmap", "correlation_heatmap.png"),
]

cols = st.columns(3)
for idx, (title, filename) in enumerate(key_plots):
    plot_path = plots_dir / filename
    with cols[idx]:
        st.subheader(title)
        if plot_path.exists():
            img = Image.open(plot_path)
            st.image(img, use_container_width=True)
        else:
            st.info(f"{filename} not found.")

st.markdown("---")
st.markdown("### 📊 Distribution Statistics")
st_cols = st.columns(2)
dist_plots = [
    ("Species Distribution", "species_distribution.png"),
    ("Biomass Distribution", "biomass_distribution.png")
]
for idx, (title, filename) in enumerate(dist_plots):
    plot_path = plots_dir / filename
    with st_cols[idx]:
        st.subheader(title)
        if plot_path.exists():
            img = Image.open(plot_path)
            st.image(img, use_container_width=True)
        else:
            st.info(f"{filename} not found.")
