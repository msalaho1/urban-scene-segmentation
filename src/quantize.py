"""
this script quantizes the DeepLabV3 model to INT8 dynamically.
"""

import torch
from torch.quantization import quantize_dynamic
from src.deeplabv3 import DeepLabV3Model

def quantize_model(weight_path, quantized_path):
    """
    Quantize the PyTorch model to INT8 dynamically.

    Args:
        weight_path (str): Path to the original model weights.
        quantized_path (str): Path to save the quantized model.
    """
    model = DeepLabV3Model(weight_path=weight_path).model
    quantized_model = quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )

    torch.save(quantized_model.state_dict(), quantized_path)
    print(f"Quantized model saved at {quantized_path}")

if __name__ == "__main__":
    weight_path = "src/weights/best_model.pth"
    quantized_path = "src/weights/quantized_model.pth"

    quantize_model(weight_path, quantized_path)