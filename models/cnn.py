from omegaconf import DictConfig
from torch import nn 


def get_activation(activation: str): 
    if activation.lower() == "linear": 
        return nn.Identity()
    elif activation.lower() == "sigmoid": 
        return nn.Sigmoid()
    elif activation.lower() == "relu": 
        return nn.ReLU()

    else: 
        raise ValueError(f"Activation {activation} not implemented. Choose between linear, sigmoid or relu")
    


class CNN_Model(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))


        self.adv_pool = nn.AdaptiveAvgPool2d(1)


        activation = get_activation(cfg["activation"])
        self.joint_training = cfg.joint_training
        if self.joint_training: 
            self.regressor_hood = nn.Sequential(nn.Linear(256, 1), activation)
            self.regressor_backdoor_left = nn.Sequential(nn.Linear(256, 1), activation)
        else: 
            self.regressor = nn.Sequential(nn.Linear(256, 1), activation)

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        out = self.adv_pool(out)
        out = out.squeeze()
        if self.joint_training: 
            out1 = self.regressor_hood(out)
            out2 = self.regressor_backdoor_left(out)
            return out1, out2 
        else: 
            out = self.regressor(out)
            return out 
        