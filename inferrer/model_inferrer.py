import sys

sys.path.append("/app")
import glob
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import os
from trainers import LightingModel
import torch
from torchvision import transforms


class Model_Inferrer:
    """
    Model Inference Class for loading pretrained weights and making predictions
    """

    def __init__(self):
        self.model = self._load_weights()

    def _load_weights(self, model_path=None):
        """
        Load pretrained weights into model
        """
        cur_dir = os.getcwd()
        os.chdir("..")
        if model_path is None:
            model_path = "arch_vit_batchsize_6_weights_imagenet_img_size_224_augmentations_False_joint_training_True_hood_training_False_activation_sigmoid"
        cfg = OmegaConf.load(f"runs_configs_hydra/{model_path}/.hydra/config.yaml")
        file = glob.glob(f"lightning_logs/{model_path}/*/checkpoints/*.ckpt")[-1]
        print(file)

        model = LightingModel(cfg)
        device = torch.device("cpu")
        checkpoint = torch.load(file, map_location=device)

        model.load_state_dict(checkpoint["state_dict"])

        os.chdir(cur_dir)
        return model

    def preprocess(self, image: str) -> torch.Tensor:
        """
        Preprocess image to the desired format
        """
        image = np.asarray(Image.open(image)).astype(np.float32)
        transform = self.model.val_transform
        sample = transform(image=image)
        image = sample["image"].unsqueeze(0)
        assert len(image.shape) == 4, "Image Dimensions are not compatible"
        assert image.shape[2] == image.shape[3] == 224, "Image Size is not compatible"

        return image

    @torch.no_grad()
    def inference(self, img=None):
        img = self.preprocess(img)
        (
            hood_perspective_predictions,
            backdoor_left_perspective_predictions,
        ) = self.model(img)
        hood_perspective_predictions = hood_perspective_predictions.squeeze().item()
        backdoor_left_perspective_predictions = (
            backdoor_left_perspective_predictions.squeeze().item()
        )
        return format(round(hood_perspective_predictions, 2)), format(
            round(backdoor_left_perspective_predictions, 2)
        )


if __name__ == "__main__":
    inferrer = Model_Inferrer()
    #img_path = r"../data/imgs/0a9a0202-d22e-447b-9be5-f922bd7d9116.jpg"
    #score1, score2 = inferrer.inference(img_path)
    #print(score1, score2)
