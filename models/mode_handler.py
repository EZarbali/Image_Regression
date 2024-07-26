from .vit import ViTRegressor
from .resnet import ResNetRegressor
from .cnn import CNN_Model


class ModeHandler:
    def __init__(self, model_name):
        self.obj = self.class_factory(model_name)

    @property
    def model(self):
        return self.obj

    def class_factory(self, model_name):
        if "vit" in model_name:
            return ViTRegressor

        elif "resnet" in model_name:
            return ResNetRegressor

        elif "cnn" in model_name:
            return CNN_Model
        else:
            raise ValueError(
                f"Model not known: {model_name} | Choose between resnet, vit and cnn"
            )
