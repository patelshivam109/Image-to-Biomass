"""
Custom Dataset for Image-to-Biomass Prediction
"""

from pathlib import Path

import pandas as pd
import torch

from PIL import Image

from torch.utils.data import Dataset

from src.config import (
    IMAGE_COLUMN,
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    TARGET_COLUMNS,
    RAW_DATA_DIR
)


class BiomassDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        # -----------------------------------
        # Load Image
        # -----------------------------------

        image_path = RAW_DATA_DIR / row[IMAGE_COLUMN]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # -----------------------------------
        # Metadata
        # -----------------------------------

        numerical_features = row[NUMERICAL_COLUMNS].values.astype("float32")

        categorical_features = row[CATEGORICAL_COLUMNS].values.astype("float32")

        metadata = torch.tensor(
            list(numerical_features) + list(categorical_features),
            dtype=torch.float32
        )

        # -----------------------------------
        # Targets
        # -----------------------------------

        targets = torch.tensor(
            row[TARGET_COLUMNS].values.astype("float32"),
            dtype=torch.float32
        )

        return {
            "image": image,
            "metadata": metadata,
            "targets": targets
        }