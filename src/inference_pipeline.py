"""
the Urban Semantic Segmentation (USS) class is the inference pipeline that includes all model's useful methods
"""

from src.load_configs import load_configs
from src.preprocessing import preprocess
from src.deeplabv3 import DeepLabV3Model
from src.postprocessing import postprocess


class USS:
    def __init__(self, config_path: str):
        """
        Initialize the USS class with the configuration file path.
        Args:
            config_path (str): Path to the configuration file.
        """
        self.config       = load_configs(config_path)
        self.weight_path  = self.config['model']['weight_path']
        self.encoder_name = self.config['model']['encoder_name']
        self.num_classes  = self.config['model']['num_classes']
        self.device       = self.config['model']['device']
        # self.class_map    = self.config['class_map']
        

    def preprocess(self, image):
        """
        Preprocess the input image for the model.
        Args:
            image (numpy.ndarray): Input image to be preprocessed.
        Returns:
            torch.Tensor: Preprocessed image tensor ready for model inference.
        """
        return preprocess(image)
    
    def predict(self, pre_image):
        """
        Run the model inference on the preprocessed image.
        Args:
            pre_image (torch.Tensor): Preprocessed image tensor.
        Returns:
            torch.tensor: Model outputs.
        """

        dlv3 = DeepLabV3Model(weight_path= self.weight_path, encoder_name= self.encoder_name, num_classes= self.num_classes, device= self.device)
        
        return dlv3.infer(pre_image)
    
    def postprocess(self, outputs):
        """
        Postprocess the model outputs to obtain the final segmentation results.
        Args:
            outputs (numpy.ndarray): Model outputs.
        Returns:
            
        """
        return postprocess(outputs)