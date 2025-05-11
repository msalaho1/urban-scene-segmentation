"""
Example of how to use the main function

"""

from src.main import main
import cv2


INPUT_IMAGE = 'data/testing/image3.png'
CONFIG_PATH = 'configs/configs.yaml'
visualize   = True

if __name__ == "__main__":

    image= cv2.imread(INPUT_IMAGE)
    results = main(image, CONFIG_PATH, visualize)
    print(f"the input image included these classes: {results}")
   
