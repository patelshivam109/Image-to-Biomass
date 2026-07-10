import streamlit as st

st.set_page_config(page_title="About", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700; font-size: 2.2rem; color: #E8F5E9; margin-bottom: 5px;
    }
    .hero-accent {
        background: linear-gradient(135deg, #FF7043, #FF5722);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem; font-weight: 700; color: #FF7043;
        text-transform: uppercase; letter-spacing: 2px; margin-bottom: 5px;
    }
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem; font-weight: 700; color: #ECEFF1; margin-bottom: 15px;
    }
    .styled-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 112, 67, 0.3), transparent);
        margin: 30px 0; border: none;
    }
    .about-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px; padding: 28px; margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .about-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(255, 112, 67, 0.25);
    }
    .about-card h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #ECEFF1; font-size: 1.15rem; margin-bottom: 12px;
    }
    .about-card p, .about-card li {
        color: #B0BEC5; font-size: 0.9rem; line-height: 1.8;
    }
    .tech-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 4px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #CFD8DC;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">About <span class="hero-accent">This Project</span></div>', unsafe_allow_html=True)
st.markdown('<div style="color: #78909C; font-size: 0.9rem; margin-bottom: 30px;">Understanding the problem, the architecture, and the technology behind Precision Pasture AI.</div>', unsafe_allow_html=True)

# --- Problem ---
st.markdown("""
<div class="about-card">
    <h3>The Problem</h3>
    <p>
        Estimating pasture biomass traditionally requires a labor-intensive process known as <strong>"clipping and weighing"</strong>. 
        A farmer must physically cut a square meter of grass, dry it in an oven to remove all moisture, 
        and weigh it on a scale. This process is repeated across the entire field to determine the 
        available feed measured in kilograms per hectare (kg/ha).
    </p>
    <p style="margin-top: 10px;">
        This application <strong>automates the entire workflow</strong> using Computer Vision and Deep Learning. 
        By uploading a single top-down RGB photograph of the pasture and providing basic environmental metadata, 
        the AI instantly predicts dry biomass yield across multiple vegetation categories.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# --- Architecture ---
st.markdown('<div class="section-label">Architecture</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">How the Model Works</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="about-card">
        <h3>EfficientNet-B0 Visual Backbone</h3>
        <p>
            The core of the model uses <strong>EfficientNet-B0</strong>, a state-of-the-art convolutional 
            neural network optimized for visual feature extraction. It analyzes the uploaded RGB image 
            to detect complex patterns including leaf texture, canopy density, color distribution, 
            and vegetation coverage.
        </p>
        <p style="margin-top: 8px;">
            EfficientNet achieves superior accuracy with significantly fewer parameters than 
            alternatives like ResNet or VGG, making it ideal for deployment on resource-constrained servers.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="about-card">
        <h3>Dual-Input Feature Fusion</h3>
        <p>
            Images alone can be misleading &mdash; the same grass looks different in winter vs. summer, 
            or in New South Wales vs. Tasmania. To solve this, the model has a <strong>second neural 
            network branch</strong> that processes environmental metadata:
        </p>
        <ul>
            <li><strong>NDVI</strong> &mdash; Normalized Difference Vegetation Index (plant health)</li>
            <li><strong>Canopy Height</strong> &mdash; Estimated or measured grass height</li>
            <li><strong>Geographic State</strong> &mdash; Regional climate influence</li>
            <li><strong>Species</strong> &mdash; Grass type (e.g., Ryegrass, Clover, Kikuyu)</li>
        </ul>
        <p style="margin-top: 8px;">
            Both branches are <strong>fused</strong> and passed through a regression head to output 
            precise continuous values in kg/ha.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# --- Visual Tools ---
st.markdown('<div class="section-label">Visual Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Transparency Tools</div>', unsafe_allow_html=True)

st.markdown("""
<div class="about-card">
    <h3>Making AI Decisions Transparent</h3>
    <p>This dashboard employs two visual analysis tools to make the AI's reasoning interpretable:</p>
    <ul>
        <li>
            <strong>HSV Semantic Segmentation</strong> &mdash; A color-space algorithm that isolates 
            green vegetation from soil, dead material, and shadows. Provides an instant binary mask 
            showing living plant matter with a vegetation coverage percentage.
        </li>
        <li style="margin-top: 8px;">
            <strong>Grad-CAM (Gradient-weighted Class Activation Mapping)</strong> &mdash; Hooks into 
            the final convolutional layers of EfficientNet during inference. By analyzing the gradients, 
            it generates a heatmap overlay showing exactly which regions of the image contributed most 
            to the biomass prediction.
        </li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# --- Tech Stack ---
st.markdown('<div class="section-label">Stack</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Technology</div>', unsafe_allow_html=True)

badges = ["PyTorch", "EfficientNet-B0", "Streamlit", "OpenCV", "scikit-learn", "Pandas", "NumPy", "Grad-CAM", "Python 3.10"]
badge_html = "".join([f'<span class="tech-badge">{b}</span>' for b in badges])
st.markdown(f'<div style="margin-bottom: 30px;">{badge_html}</div>', unsafe_allow_html=True)

st.markdown("""
<div style="color: #546E7A; font-size: 0.82rem; text-align: center; padding: 20px 0;">
    Built for Precision Agriculture &bull; Powered by Deep Learning
</div>
""", unsafe_allow_html=True)
