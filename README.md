# Image-to-Biomass Prediction Using Computer Vision and Deep Learning....

An AI-powered precision agriculture solution designed to estimate pasture biomass directly from RGB images captured via smartphones, drones, or field cameras. This system eliminates the need for time-consuming manual harvesting, drying, and weighing by leveraging advanced semantic segmentation, depth estimation, and regression models.

---

## 📌 Project Overview

Traditional biomass measurement methods are highly labor-intensive and costly. This project automates the process by analyzing pasture images to provide accurate, real-time biomass estimations ($kg/ha$).

### Core Pipeline
1. **Semantic Segmentation:** Uses models like U-Net or DeepLabV3+ to segment vegetation categories (grass, clover, dead plant material).
2. **Feature Extraction:** Analyzes visual characteristics such as vegetation coverage, texture, color distribution, and canopy density.
3. **Depth Estimation (Optional):** Integrates MiDaS to capture canopy height and structural data from monocular images.
4. **Biomass Regression:** Predicts specific biomass components and generates visual heatmaps highlighting high-yield regions.

---

## 📊 Target Metrics & Predictions

The system provides precise numerical estimations (measured in **kg/ha**) for the following parameters:

* **Dry Green Vegetation Biomass** (excluding clover)
* **Dry Dead Material Biomass**
* **Dry Clover Biomass**
* **Green Dry Matter (GDM)**
* **Total Dry Biomass**

---

## 🛠️ Technologies Used

| Category | Tools & Technologies |
| :--- | :--- |
| **Core Language** | Python |
| **Computer Vision** | OpenCV, MiDaS (Depth Estimation) |
| **Deep Learning Frameworks** | PyTorch / TensorFlow |
| **Model Architectures** | U-Net, DeepLabV3+ |
| **Machine Learning & Data** | Scikit-learn, NumPy, Pandas, Matplotlib |
| **Deployment / UI** | Streamlit / Flask |

---

## 🚀 Expected Outcomes

* **Automated Estimation:** Rapid biomass calculation directly from field images.
* **Real-Time Monitoring:** Instant insights into pasture health and growth.
* **Visual Heatmaps:** Clear visualization of density and biomass distribution.
* **Data-Driven Decisions:** Optimized grazing schedules, forage availability tracking, and sustainable feed management.
* **Cost Efficiency:** Drastic reduction in manual labor and field survey costs.

---

## 📁 Repository Structure

```text
├── data/               # Placeholder for sample images and datasets
├── models/             # Model architectures (Segmentation, Regression, Depth)
├── notebooks/          # Exploratory Data Analysis and training experiments
├── src/                # Core source code for processing and inference
├── app.py              # Streamlit / Flask application entry point
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
