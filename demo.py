"""
Example of how to use the main function

"""

from src.main import main


INPUT_IMAGE = 'data/testing/image1.jpeg'
CONFIG_PATH = 'configs/configs.yaml'
visualize   = True

if __name__ == "__main__":

    results = main(INPUT_IMAGE, CONFIG_PATH, visualize)
    print(f"the input image included these classes: {results}")
   
