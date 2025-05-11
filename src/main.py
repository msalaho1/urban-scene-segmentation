"""
this script includes the main function that runs the inference pipeline
"""


from typing import Any
from src.inference_pipeline import USS
from src.utils import draw_mask
from src.logger_setup import setup_logger
import time
import pdb

# Initialize logger
logger = setup_logger("inference", "./logs/app.log")

def main(image, config_path: str, visualize: bool= False)-> Any:
    """
    Main function to run the inference pipeline.
    Args:
        image_path (str): Path to the input image.
        config_path (str): Path to the configuration file.
        visualize (bool): Flag to visualize the results. Default is False.
    """
    start_time = time.time()

    pipline  = USS(config_path= config_path)

    if image is None:
        logger.error("Image not found at image_path.")
        raise ValueError(f"Image not found at image_path")

    try:
        pre_image = pipline.preprocess(image)
        outputs = pipline.predict(pre_image)
        pred_masks, pred_classes = pipline.postprocess(outputs)
      
        if visualize:
            logger.info("Visualizing and saving the output mask.")
            draw_mask(image, pred_masks, './output/output.jpg')

    except Exception as e:
        logger.error(f"An error occurred during inference: {e}")
        return None

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds.")

    return pred_classes
