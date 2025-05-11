"""
this script includes the main function that runs the inference pipeline
"""


from typing import Any
from src.inference_pipeline import USS
from src.utils import draw_mask
import pdb

def main(image, config_path: str, visualize: bool= False)-> Any:
    """
    Main function to run the inference pipeline.
    Args:
        image_path (str): Path to the input image.
        config_path (str): Path to the configuration file.
        visualize (bool): Flag to visualize the results. Default is False.
    """
    pipline  = USS(config_path= config_path)

    if image is None:
        raise ValueError(f"Image not found at image_path")

    try:
        pre_image                = pipline.preprocess(image)
        outputs                  = pipline.predict(pre_image)
        pred_masks, pred_classes = pipline.postprocess(outputs)

        if visualize:
            draw_mask(image, pred_masks, './output/output.jpg')

    except Exception as e:
        print(f"An error occurred during inference: {e}")

        return None
    
    return pred_classes
