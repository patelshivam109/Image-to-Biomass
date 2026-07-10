import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import json
from PIL import Image
from src.config import BIOMASS_G_TO_KG_HA

st.set_page_config(page_title="Model Metrics", layout="wide")

# Shared CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700; font-size: 2.2rem; color: #E8F5E9;
        margin-bottom: 5px;
    }
    .hero-accent {
        background: linear-gradient(135deg, #42A5F5, #26C6DA);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem; font-weight: 700; color: #42A5F5;
        text-transform: uppercase; letter-spacing: 2px; margin-bottom: 5px;
    }
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem; font-weight: 700; color: #ECEFF1; margin-bottom: 15px;
    }
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px; padding: 16px 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        border-color: rgba(66, 165, 245, 0.4);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="metric-container"] label {
        color: #90A4AE !important; font-size: 0.8rem !important;
        text-transform: uppercase; letter-spacing: 1px;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #E8F5E9 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
    }
    .styled-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(66, 165, 245, 0.3), transparent);
        margin: 30px 0; border: none;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px; padding: 24px; margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">Model <span class="hero-accent">Training Analytics</span></div>', unsafe_allow_html=True)
st.markdown('<div style="color: #78909C; font-size: 0.9rem; margin-bottom: 30px;">Review the regression model performance, learning curves, and evaluation plots.</div>', unsafe_allow_html=True)

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
    
    # Key Metrics
    final_train = history['train_loss'][-1]
    final_val = history['val_loss'][-1]
    best_val = min(history['val_loss'])
    best_epoch = history['val_loss'].index(best_val) + 1
    rmse = (best_val ** 0.5) * BIOMASS_G_TO_KG_HA
    
    st.markdown('<div class="section-label">Performance Summary</div>', unsafe_allow_html=True)
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best Validation Loss", f"{best_val:.2f}", f"Epoch {best_epoch}")
    k2.metric("Final Training Loss", f"{final_train:.2f}")
    k3.metric("Final Validation Loss", f"{final_val:.2f}")
    k4.metric("RMSE (Accuracy)", f"±{rmse:.1f} kg/ha")
    
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    
    # Learning Curve
    st.markdown('<div class="section-label">Learning Curve</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Training vs Validation Loss</div>', unsafe_allow_html=True)
    st.line_chart(df, height=380, use_container_width=True)
    
else:
    st.warning("Training history not found.")

st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# --- EVALUATION PLOTS ---
st.markdown('<div class="section-label">Evaluation</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Model Evaluation Plots</div>', unsafe_allow_html=True)

plots_dir = PROJECT_ROOT / "outputs" / "plots"

key_plots = [
    ("Actual vs Predicted", "actual_vs_predicted.png"),
    ("Residual Analysis", "residual_plot.png"),
    ("Feature Correlations", "correlation_heatmap.png"),
]

cols = st.columns(3)
for idx, (title, filename) in enumerate(key_plots):
    plot_path = plots_dir / filename
    with cols[idx]:
        if plot_path.exists():
            img = Image.open(plot_path)
            st.image(img, caption=title, use_container_width=True)
        else:
            st.info(f"{filename} not found.")

st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Data Distribution</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Dataset Analysis</div>', unsafe_allow_html=True)

st_cols = st.columns(2)
dist_plots = [
    ("Species Distribution", "species_distribution.png"),
    ("Biomass Distribution", "biomass_distribution.png")
]
for idx, (title, filename) in enumerate(dist_plots):
    plot_path = plots_dir / filename
    with st_cols[idx]:
        if plot_path.exists():
            img = Image.open(plot_path)
            st.image(img, caption=title, use_container_width=True)
        else:
            st.info(f"{filename} not found.")
