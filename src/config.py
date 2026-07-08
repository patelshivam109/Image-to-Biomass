from pathlib import Path

# -----------------------------
# Project Directories
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "datasets" / "processed"

TRAIN_CSV = PROCESSED_DATA_DIR / "train_processed.csv"
VAL_CSV = PROCESSED_DATA_DIR / "val_processed.csv"

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
REPORT_DIR = OUTPUT_DIR / "reports"

# -----------------------------
# Image Settings
# -----------------------------
IMAGE_SIZE = 224

# -----------------------------
# Training Parameters
# -----------------------------
BATCH_SIZE = 16
NUM_WORKERS = 2

LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
RANDOM_STATE = 42

# -----------------------------
# Dataset Columns
# -----------------------------
IMAGE_COLUMN = "image_path"

NUMERICAL_COLUMNS = [
    "NDVI",
    "Height_cm"
]

CATEGORICAL_COLUMNS = [
    "State",
    "Species"
]

TARGET_COLUMNS = [
    "Dry_Green_g",
    "Dry_Dead_g",
    "Dry_Clover_g",
    "Dry_Total_g",
    "GDM_g"
]