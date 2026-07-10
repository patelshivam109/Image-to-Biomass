import torch

class GradCam:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Extracts gradients and activations from the final convolutional layer of a PyTorch model
    to generate visual heatmaps explaining the model's predictions.
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hook into the last convolutional layer of EfficientNet
        target_layer = self.model.feature_extractor[-1]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, img_tensor, meta_tensor, target_idx=0): 
        output = self.model(img_tensor, meta_tensor)
        self.model.zero_grad()
        score = output[0, target_idx]
        score.backward(retain_graph=True)
        
        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (gradients * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().detach().cpu().numpy()
