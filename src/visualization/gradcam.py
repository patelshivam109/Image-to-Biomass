"""
Grad-CAM Visualization
"""

import torch
import cv2
import numpy as np

from PIL import Image


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(
            self.save_activation
        )

        self.target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):

        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]

    def generate(self, image_tensor):

        self.model.eval()

        output = self.model(image_tensor)

        score = output.max()

        self.model.zero_grad()

        score.backward()

        gradients = self.gradients[0]

        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32
        )

        for i, w in enumerate(weights):

            cam += w * activations[i]

        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()

        cam = cv2.resize(
            cam,
            (224, 224)
        )

        cam = cam - cam.min()

        cam = cam / (cam.max() + 1e-8)

        return cam