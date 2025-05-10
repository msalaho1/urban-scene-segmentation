import cv2
import numpy as np
import pandas as pd

def save_img(img, path)-> None:
    """
    Save an image to a specified path.
    
    Parameters:
    img (numpy.ndarray): The image to save.
    path (str): The file path where the image will be saved.
    """

    cv2.imwrite(path, img)

def draw_mask(img: np.ndarray, mask: np.ndarray, output_img) -> np.ndarray:
    """
    Draw a mask on an image.

    Parameters:
    img (numpy.ndarray): The original image (H, W, 3).
    mask (numpy.ndarray): The predicted mask (H, W).

    Returns:
    numpy.ndarray: The image with the mask drawn on it.
    """
    resized= cv2.resize(img, (640, 640))
    mask_colored = cv2.applyColorMap((mask * 10).astype(np.uint8), cv2.COLORMAP_JET)
    img_with_mask = cv2.addWeighted(resized, 0.7, mask_colored, 0.3, 0)
    save_img(img_with_mask, output_img)

def load_class_map()-> dict:
    """
    Load the class map from a CSV file.
    
    Returns:
    dict: The class map dictionary.
    """
    class_map = pd.read_csv('./data/classes.csv')
    class_map.columns = ['class_id', 'class_name']
    id2label = dict(zip(class_map['class_id'], class_map['class_name']))
   
    return id2label
