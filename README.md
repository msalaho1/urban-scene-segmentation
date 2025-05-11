# Urban Scene Segmentation

![Urban Scene Segmentation Cover](sample_output.jpg)

## Introduction
Urban Scene Segmentation is a critical task in computer vision, enabling the identification and classification of various objects in urban environments. This project implements an urban scene segmentation pipeline using the DeepLabV3 model, providing preprocessing, model inference, and postprocessing steps, along with an API for serving predictions.

## Why DeepLabV3?
DeepLabV3 is a state-of-the-art architecture for semantic segmentation. It uses atrous convolution to capture multi-scale context and provides high accuracy for dense prediction tasks like urban scene segmentation. This makes it an excellent choice for this project.

## Dataset Notes
- The dataset includes 149 images (112 training and 37 validation).
- Masks are saved as PNG files with the same name as the corresponding images but ending with `_mask`.
- Preprocessing applied:
  - Auto-orientation of pixel data (with EXIF-orientation stripping).
  - Resize to 640x640 (Stretch).
- No image augmentation techniques were applied.

## Project Structure
- **`api.py`**: FastAPI application for serving the model.
- **`demo.py`**: Example script to run the segmentation pipeline.
- **`src/`**: Contains core modules like preprocessing, model definition, and utilities.
- **`configs/`**: Configuration files.
- **`data/`**: Input images and related data.
- **`weights/`**: Pretrained and quantized model weights.

## Model Architecture
The project uses the DeepLabV3 architecture with a MobileNetV2 encoder. This combination provides a balance between accuracy and efficiency, making it suitable for urban scene segmentation tasks.

## Architecture Trials
1. **Baseline Model**:
   - Started with a simple U-Net architecture.
   - Achieved moderate accuracy but was computationally expensive.

2. **DeepLabV3 with ResNet Encoder**:
   - Improved accuracy but increased model size and inference time.

3. **DeepLabV3 with MobileNetV2 Encoder**:
   - Provided the best balance of accuracy and efficiency.

## used techniques in training:
- **Learning Rate Scheduler**: Used `ReduceLROnPlateau` to dynamically adjust the learning rate based on validation loss.
- **Early Stopping**: Implemented early stopping to terminate training when no improvement was observed for 5 consecutive epochs.
- **Data Augmentation**: Applied Gaussian blur and normalization during preprocessing to improve generalization.
- **Visualization**: Visualized predictions and overlays to qualitatively assess model performance.
- **Quantization**: Reduced model size and improved inference speed using dynamic quantization.
- **ONNX Conversion**: Enabled deployment in diverse environments by converting the model to ONNX format.

## Why PyTorch?
PyTorch was chosen for its:
- Extensive library support for state-of-the-art models like DeepLabV3.
- straight forward onnx conversion and quantization processes.

## Setup and Installation

### Installation
1. Clone the repository:
   ```bash
   git clone <https://github.com/msalaho1/urban-scene-segmentation.git>
   cd urban-scene-segmentation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the required Python version (3.10).

### Using the API
1. Start the API server:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

2. Access the health check endpoint:
   ```bash
   curl http://localhost:8000/health
   ```

3. Use the `/predict/` endpoint to upload an image and get predictions.

   Example using `curl`:
   ```bash
   curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@data/testing/image1.jpeg"
   ```

### Using Docker
1. Build the Docker image:
   ```bash
   docker build -t uss .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 -v $(pwd)/output:/app/output uss
   ```

3. Access the API as described above.

### Quantization and ONNX Conversion

1. **Quantization**:
   Quantization reduces the model size and speeds up inference. The script `src/quantize.py` handles this process. Run:
   ```bash
   python -m src.quantize.py
   ```

2. **ONNX Conversion**:
   Convert the PyTorch model to ONNX format for deployment. Use the script `src/conv2onnx.py`:
   ```bash
   python -m src.conv2onnx.py
   ```

   The converted models are saved in the `src/weights/` directory.

## Example Usage

Run the demo script to test the pipeline:
```bash
python demo.py
```

This will process the input image and display the segmentation results.

## Trials

### Data Investigation
Before training, the dataset was thoroughly investigated to understand its structure and characteristics. Key findings include:
- The dataset contains 149 images (112 for training and 37 for validation).
- Images were resized to 640x640 with no augmentation applied.
- Masks are stored in the same folder as the images, with filenames ending in `_mask`.
- Class mappings were verified to ensure consistency between training and validation datasets.

### Experimentation

1. **UNet with ResNet-34 Encoder**:
   - Initial trials used UNet with a ResNet-34 encoder.
   - Results showed moderate performance but room for improvement.

2. **UNet with MobileNetV2 Encoder**:
   - Switched to MobileNetV2 as the encoder for UNet to reduce model size and improve inference speed.
   - Performance improved slightly, but training was still resource-intensive.

3. **Batch Size Reduction**:
   - Reduced the batch size to fit the model into GPU memory.
   - This allowed training to proceed without memory errors but increased training time.

4. **Input Image Size Reduction**:
   - Decreased input image sizes to further optimize memory usage.
   - This led to faster training but slightly reduced accuracy.

5. **Mixed Precision Training**:
   - Enabled mixed precision training using PyTorch's `autocast` and `GradScaler`.
   - This significantly reduced memory usage and improved training speed.

6. **Weight Decay**:
   - Added weight decay to the optimizer to regularize the model and prevent overfitting.

7. **DeepLabV3 with MobileNetV2 Encoder**:
   - Finally, switched to DeepLabV3 with a MobileNetV2 encoder.
   - This architecture provided the best balance of accuracy and efficiency.

### Additional Techniques
- **Learning Rate Scheduler**: Used `ReduceLROnPlateau` to dynamically adjust the learning rate based on validation loss.
- **Early Stopping**: Implemented early stopping to terminate training when no improvement was observed for 5 consecutive epochs.
- **Data Augmentation**: Applied Gaussian blur and normalization during preprocessing to improve generalization.
- **Visualization**: Visualized predictions and overlays to qualitatively assess model performance.

These trials and techniques collectively contributed to the final model's performance and efficiency.

## Logs
Logs are saved in the `logs/` directory for debugging and monitoring.
