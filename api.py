"""
this script is used to create a FastAPI application that serves the USS model.
It includes an endpoint to upload an image and receive a predicted segmentation mask in response.
"""

import numpy as np
import cv2
import uvicorn

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

from src.main import main

app = FastAPI()

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the service is running.
    Returns:
        JSONResponse: A simple message indicating the service status.
    """
    return JSONResponse(content={"status": "ok"})


@app.post("/predict/")
async def predict(file: UploadFile = File(...), visualize: bool = Form(False)):
    """
    Endpoint to predict segmentation mask for an uploaded image.

    Args:
        file (UploadFile): Uploaded image file.
        visualize (bool): Flag to enable visualization. Default is False.

    Returns:
        JSONResponse: Predicted classes or visualization output.
    """
    try:
        contents = await file.read()
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        pred_classes = main(image=image, config_path="configs/configs.yaml", visualize=visualize)

        if visualize:
            return JSONResponse(content={"message": "Visualization saved as output.jpg", "classes": pred_classes})
        else:
            return JSONResponse(content={"classes": pred_classes})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)