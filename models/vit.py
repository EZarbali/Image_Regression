
from omegaconf import DictConfig

import torch 
from torch import nn 
from transformers import ViTModel 

def get_activation(activation: str): 
    if activation.lower() == "linear": 
        return nn.Identity()
    elif activation.lower() == "sigmoid": 
        return nn.Sigmoid()
    elif activation.lower() == "relu": 
        return nn.ReLU()

    else: 
        raise ValueError(f"Activation {activation} not implemented. Choose between linear, sigmoid or relu")


class ViTRegressor(nn.Module): 

    def __init__(self, cfg: DictConfig): 
        super().__init__()
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
    
        if cfg.finetune: 
            for p in self.backbone.parameters(): 
                p.requires_grad = False
        activation = get_activation(cfg["activation"])
        self.joint_training = cfg.joint_training
        if self.joint_training: 

            if cfg.projection_head: 
                self.regressor_hood = nn.Sequential(nn.Linear(self.backbone.config.hidden_size, 256), 
                                                    nn.ReLU(),
                                                    nn.Dropout(0.5),
                                                    nn.Linear(256, 1), 
                                                    activation)
                self.regressor_backdoor_left = nn.Sequential(nn.Linear(self.backbone.config.hidden_size, 256), 
                                                    nn.ReLU(),
                                                    nn.Dropout(0.5),
                                                    nn.Linear(256, 1), 
                                                    activation)
            
            else:
                self.regressor_hood = nn.Sequential(nn.Linear(self.backbone.config.hidden_size, 1), activation) 
                self.regressor_backdoor_left = nn.Sequential(nn.Linear(self.backbone.config.hidden_size, 1), activation)
        else: 
            self.regressor = nn.Sequential(nn.Linear(self.backbone.config.hidden_size, 1), activation)

    
    def forward(self, x): 
        out = self.backbone(pixel_values = x)
        # take only cls token 
        out = out.last_hidden_state[:, 0, :]
        if self.joint_training: 
            out1 = self.regressor_hood(out)
            out2 = self.regressor_backdoor_left(out)
            return out1, out2 
        else: 
            out = self.regressor(out)
            return out 
