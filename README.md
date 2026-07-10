# Image-to-Biomass Prediction Using Computer Vision and Deep Learning....

An AI-powered precision agriculture solution designed to estimate pasture biomass directly from RGB images captured via smartphones, drones, or field cameras. This system eliminates the need for time-consuming manual harvesting, drying, and weighing by leveraging advanced semantic segmentation, depth estimation, and regression models.

----

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

## 🚀 How to Run the Dashboard

1. **Set up your environment:**
   Create a virtual environment and activate it:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install Dependencies:**
   Install the required libraries (Note: requires NumPy < 2.0 to avoid PyTorch/OpenCV conflicts):
   ```powershell
   pip install -r requirements.txt
   pip install "numpy<2.0" "opencv-python<4.11"
   ```

3. **Run the Streamlit App:**
   ```powershell
   python -m streamlit run dashboard/app.py
   ```

---

## 📁 Repository Structure

```text
IMAGE_TO_BIOMASS
│
├── dashboard/
│   └── app.py                  # Streamlit application entry point
│
├── datasets/
│   ├── raw/                    # Raw field images and CSVs
│   └── processed/              # Cleaned datasets
│
├── models/
│   ├── best_model.pth          # Trained EfficientNet model weights
│   ├── scaler.pkl              # Data normalizer
│   └── label_encoders.pkl      # Categorical encoders
│
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_training.ipynb
│   └── 05_evaluation.ipynb
│
├── outputs/
│   ├── plots/
│   ├── predictions/
│   └── reports/
│
├── src/
│   ├── data/                   # Data loaders and transforms
│   ├── models/                 # PyTorch architectures (Biomass, Segmentation)
│   ├── utils/                  # Training scripts and metrics
│   └── config.py               # Global configurations
│
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```
