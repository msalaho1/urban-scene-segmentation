import torch
import numpy as np
from typing import Any
from src.utils import load_class_map

def postprocess(model_output: torch.Tensor) -> tuple[np.ndarray, list[str]]:
    """
    Post-process the model output to generate the predicted mask and extract unique classes.

    Args:
        model_output (torch.Tensor): Model output tensor of shape (1, C, H, W), where C is the number of classes.

    Returns:
        tuple[np.ndarray, list[int]]: A tuple containing:
            - pred_mask (np.ndarray): Predicted mask as a NumPy array of shape (H, W).
            - unique_classes (list[str]): List of unique class indices present in the predicted mask.
    """
    pred_mask       = torch.argmax(model_output.squeeze(), dim=0).cpu().numpy()
    unique_classes  = list(np.unique(pred_mask))
    class_map       = load_class_map()
    pred_classes    = [class_map.get(i, i) for i in unique_classes]

    return pred_mask, pred_classes