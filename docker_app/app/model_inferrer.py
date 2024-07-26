import sys 
sys.path.append("/app")
import glob 
import numpy as np 
from PIL import Image 
from omegaconf import OmegaConf
import os 
from models import ModeHandler
import torch 
from torchvision import transforms 
from typing import Tuple

class Model_Inferrer: 

    def __init__(self): 

        self.model = self._load_weights()




    def _load_weights(self, model_path=None): 
        """
        Load pretrained weights into model 
        """
        if model_path is None: 
            model_path = "arch_vit_batchsize_6_weights_imagenet_img_size_224_augmentations_False_joint_training_True_hood_training_False_activation_sigmoid"
        cfg = OmegaConf.load(f"runs_configs_hydra/{model_path}/.hydra/config.yaml")
        #file = glob.glob(f"{model_path}/*/checkpoints/*.ckpt")[-1]

        mode_handler = ModeHandler(cfg.arch)
        model = mode_handler.model

        model = model(cfg)
        
        device = torch.device('cpu')
        model.regressor_backdoor_left.load_state_dict(torch.load("regressor_backdoor_left_weights.pth", map_location=device))
        model.regressor_hood.load_state_dict(torch.load("regressor_hood_weights.pth", map_location=device))

        
        return model 


    def preprocess(self, image: str) -> torch.Tensor: 
        
        image = Image.open(image)
        transform = transforms.Compose([transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])])
        image = transform(image)
        image = image.unsqueeze(0)
        assert len(image.shape) == 4, "Image Dimensions are not compatible"
        assert image.shape[2] == image.shape[3] == 224, "Image Size is not compatible"

        return image 
    

    @torch.no_grad()
    def inference(self, img: str) -> Tuple: 

        img = self.preprocess(img)
        hood_perspective_predictions, backdoor_left_perspective_predictions = self.model(img)
        hood_perspective_predictions = hood_perspective_predictions.squeeze().item()
        backdoor_left_perspective_predictions = backdoor_left_perspective_predictions.squeeze().item()
        return format(round(hood_perspective_predictions, 2)), format(round(backdoor_left_perspective_predictions, 2))
    



if __name__ == '__main__': 

    inferrer = Model_Inferrer()
    img_path = r"image.png"
    score1,score2 = inferrer.inference(img_path)
    print(score1, score2)

