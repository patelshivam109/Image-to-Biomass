"""
Biomass Prediction Model

Image + Metadata Fusion Network
"""

import torch
import torch.nn as nn

from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights


class BiomassModel(nn.Module):

    def __init__(self):

        super().__init__()

        # --------------------------------------------------
        # Image Backbone
        # --------------------------------------------------

        backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        self.feature_extractor = backbone.features

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        image_feature_size = 1280

        # --------------------------------------------------
        # Metadata Branch
        # --------------------------------------------------

        self.metadata_branch = nn.Sequential(

            nn.Linear(5, 32),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(32, 32),

            nn.ReLU()

        )

        # --------------------------------------------------
        # Fusion Head
        # --------------------------------------------------

        self.regressor = nn.Sequential(

            nn.Linear(image_feature_size + 32, 512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, 256),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256, 5)

        )

    def forward(self, image, metadata):

        image = self.feature_extractor(image)

        image = self.avgpool(image)

        image = torch.flatten(image, 1)

        metadata = self.metadata_branch(metadata)

        fused = torch.cat([image, metadata], dim=1)

        output = self.regressor(fused)

        return output