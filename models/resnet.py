from omegaconf import DictConfig

import torch
from torch import nn

from torchvision.models import resnet18, resnet34, resnet50


def get_activation(activation: str):
    if activation.lower() == "linear":
        return nn.Identity()
    elif activation.lower() == "sigmoid":
        return nn.Sigmoid()
    elif activation.lower() == "relu":
        return nn.ReLU()

    else:
        raise ValueError(
            f"Activation {activation} not implemented. Choose between linear, sigmoid or relu"
        )


def get_resnet(model_name: str):
    if model_name.lower() == "resnet18":
        return resnet18
    elif model_name.lower() == "resnet34":
        return resnet34
    elif model_name.lower() == "resnet50":
        return resnet50

    else:
        raise ValueError(
            f"Model {model_name} not implemented. Choose between resnet18, resnet34 or resnet50"
        )


class ResNetRegressor(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        backbone = get_resnet(cfg.arch)

        if cfg.encoder_weights == "imagenet":
            backbone = backbone(weights="IMAGENET1K_V1")
        else:
            backbone = backbone(weights=None)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        if cfg.finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False

        activation = get_activation(cfg["activation"])
        self.joint_training = cfg.joint_training
        if self.joint_training:
            self.regressor_hood = nn.Sequential(nn.Linear(512, 1), activation)
            self.regressor_backdoor_left = nn.Sequential(nn.Linear(512, 1), activation)
        else:
            self.regressor = nn.Sequential(nn.Linear(512, 1), activation)

    def forward(self, x):
        out = self.backbone(x)
        # out: (Batch, 512, 1, 1) --> (Batch, 512)
        out = out.squeeze()
        if self.joint_training:
            out1 = self.regressor_hood(out)
            out2 = self.regressor_backdoor_left(out)
            return out1, out2
        else:
            out = self.regressor(out)
            return out
