import torch
import onnxruntime as ort
import numpy as np
from src.deeplabv3 import DeepLabV3Model

def convert_to_onnx(weight_path:str, onnx_path:str, input_size=(1, 3, 640, 640))-> None:
    """
    Convert the PyTorch model to ONNX format.

    Args:
        weight_path (str): Path to the PyTorch model weights.
        onnx_path (str): Path to save the ONNX model.
        input_size (tuple): Input size for the model (batch_size, channels, height, width).
    """
    model = DeepLabV3Model(weight_path=weight_path)
    model.model.eval()

    dummy_input = torch.randn(*input_size, device=model.device)

    torch.onnx.export(
        model.model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model has been converted to ONNX and saved at {onnx_path}")

def batch_inference_onnx(onnx_path, batch_images)->torch.Tensor:
    """
    Perform batch inference using the ONNX model.

    Args:
        onnx_path (str): Path to the ONNX model.
        batch_images (np.ndarray): Batch of preprocessed images (batch_size, 3, height, width).

    Returns:
        np.ndarray: Batch of predicted masks.
    """
    
    ort_session = ort.InferenceSession(onnx_path)

    ort_inputs = {ort_session.get_inputs()[0].name: batch_images}
    ort_outs = ort_session.run(None, ort_inputs)

    pred_masks = np.argmax(ort_outs[0], axis=1)
    return pred_masks

if __name__ == "__main__":
    weight_path = "src/weights/deeplabv3_quantized.pth"
    onnx_path = "src/weights/deeplabv3_quantized.onnx"

    convert_to_onnx(weight_path, onnx_path)

    batch_images = np.random.rand(4, 3, 640, 640).astype(np.float32)
    pred_masks = batch_inference_onnx(onnx_path, batch_images)
    print("Batch inference completed. Predicted masks shape:", pred_masks.shape)