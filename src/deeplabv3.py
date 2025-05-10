"""
the DeepLabV3 model class for semantic segmentation
This class is responsible for loading the model weights, performing inference on input images,
and returning the predicted segmentation masks.
"""
import torch
import segmentation_models_pytorch as smp

class DeepLabV3Model:
    def __init__(self, weight_path, encoder_name='mobilenet_v2', num_classes=87, device='cuda')->None:
        """
        Initialize the DeepLabV3 model for inference.

        Args:
            weight_path (str): Path to the model weights file.
            encoder_name (str): Name of the encoder backbone.
            num_classes (int): Number of output classes.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes
        )
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

    def infer(self, image_tensor):
        """
        Perform inference on a single image tensor.

        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor of shape (1, 3, H, W).

        Returns:
            torch.Tensor: Predicted mask of shape (H, W).
        """
        with torch.no_grad():
            output = self.model(image_tensor)  
        return output