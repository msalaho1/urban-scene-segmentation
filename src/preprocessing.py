"""
Preprocessing module for image data.
This module provides functions to preprocess images for deep learning models.
The preprocessing includes resizing, color conversion, blurring, normalization, and conversion to a PyTorch tensor.
"""

import cv2
import numpy as np
import torch

def preprocess(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess the input image for the model.

    This function performs the following steps:
    1. Converts the image from BGR to RGB color space.
    2. Resizes the image to 640x640.
    3. Normalizes the pixel values to the range [-1, 1].
    4. Converts the processed image to a PyTorch tensor.

    Args:
        image (numpy.ndarray): Input image to be preprocessed.

    Returns:
        torch.Tensor: Preprocessed image tensor ready for model inference.
    """
    rgb= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (640, 640))
    normalized = (resized / 255.0 - 0.5) / 0.5
    pre_image = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
    tensor_image = torch.tensor(pre_image).unsqueeze(0)

    return tensor_image