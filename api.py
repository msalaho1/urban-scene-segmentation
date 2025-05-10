"""
this script is used to create a FastAPI application that serves a DeepLabV3 model for image segmentation.
It includes an endpoint to upload an image and receive a predicted segmentation mask in response.
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import torch
from deeplabv3 import DeepLabV3Model
from preprocessing import preprocessing
from sr.logger_setup import setup_logger


setup_logger("api_logger", "logs/api.log")

app = FastAPI()

model = DeepLabV3Model(weight_path="src/weights/best_model.pth")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict segmentation mask for an uploaded image.

    Args:
        file (UploadFile): Uploaded image file.

    Returns:
        JSONResponse: Predicted mask as a list of class indices.
    """
    try:
        contents = await file.read()
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        preprocessed_image = preprocessing(image)
        preprocessed_image = torch.tensor(preprocessed_image).unsqueeze(0)

        pred_mask = model.infer(preprocessed_image)

        pred_mask_list = pred_mask.numpy().tolist()

        return JSONResponse(content={"predicted_mask": pred_mask_list})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)