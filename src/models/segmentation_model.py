"""
Semantic Segmentation Model (DeepLabV3+)
Architecture code to satisfy project requirements for vegetation segmentation.
"""

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

class VegetationSegmentationModel(nn.Module):
    def __init__(self, num_classes=3):
        """
        Initializes the DeepLabV3 model for pasture segmentation.
        Args:
            num_classes: Number of target classes (e.g., Grass, Clover, Dead Material)
        """
        super().__init__()
        
        # Load a pre-trained DeepLabV3 architecture
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        
        # Replace the classifier head for our specific number of classes
        # DeepLabV3 classifier is an DeepLabHead instance
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
        # Auxiliary classifier (used during training)
        self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        """
        Forward pass.
        Returns the main segmentation mask logits.
        """
        return self.model(x)['out']
