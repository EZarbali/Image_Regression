import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.utils.data import DataLoader


from .utils import get_validation_augmentations, get_training_augmentations
from dataset import CarDataset
from models import ModeHandler


class LightingModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        self.arch = self.args.arch
        model_handler = ModeHandler(self.arch)

        if self.args.use_augmentations:
            self.transform = get_training_augmentations(cfg)

        else:
            self.transform = get_validation_augmentations(cfg)
        self.val_transform = get_validation_augmentations(cfg)

        if self.arch.lower() not in {"vit", "resnet18", "resnet34", "resnet50", "cnn"}:
            raise Exception("Model Name is not implemented yet")
        else:
            model = model_handler.model

        self.model = model(cfg)
        self.loss_fn = nn.MSELoss()

        # init stacks for loggers
        self.train_losses = []
        self.train_ious = []
        self.val_losses = []
        self.val_ious = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, y_hood, y_backdoor_left = batch
        prediction = self(img)
        if self.args.joint_training:
            pred1, pred2 = prediction
            loss = self.loss_fn(pred1, y_hood) + self.loss_fn(pred2, y_backdoor_left)
        elif not self.args.joint_training and self.args.hood_training:
            loss = self.loss_fn(prediction, y_hood)
        else:
            loss = self.loss_fn(prediction, y_backdoor_left)
        self.train_losses.append(loss)
        return loss

    def on_train_epoch_end(self):
        avg_train_loss = torch.stack(self.train_losses).mean()
        self.log(
            "avg_train_loss",
            avg_train_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.train_losses.clear()

    def validation_step(self, batch, batch_idx):
        img, y_hood, y_backdoor_left = batch
        prediction = self(img)
        if self.args.joint_training:
            pred1, pred2 = prediction
            loss = self.loss_fn(pred1, y_hood) + self.loss_fn(pred2, y_backdoor_left)
        elif not self.args.joint_training and self.args.hood_training:
            loss = self.loss_fn(prediction, y_hood)

        else:
            loss = self.loss_fn(prediction, y_backdoor_left)
        self.val_losses.append(loss)
        return loss

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.val_losses).mean()
        self.log(
            "avg_val_loss", avg_val_loss, on_epoch=True, on_step=False, sync_dist=True
        )
        print(f"Vak Loss is {avg_val_loss}")
        self.val_losses.clear()

    def test_step(self, batch, batch_idx):
        img, y_hood, y_backdoor_left = batch
        prediction = self(img)
        if self.args.joint_training:
            pred1, pred2 = prediction
            loss = self.loss_fn(pred1, y_hood) + self.loss_fn(pred2, y_backdoor_left)
        elif not self.args.joint_training and self.args.hood_training:
            loss = self.loss_fn(prediction, y_hood)

        else:
            loss = self.loss_fn(prediction, y_backdoor_left)
        rmse = torch.sqrt(loss)
        self.log("RMSE_Metric", rmse, sync_dist=True)
        self.log("Loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.weight_decay,
        )
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @property
    def num_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _create_dataloader(self, mode: str, **kwargs) -> DataLoader:
        split = self._dataset_handler(mode)

        dataset = CarDataset(split, self.transform)
        return DataLoader(
            dataset, self.args.batch_size, shuffle=True, pin_memory=self.args.pin_memory
        )

    def _dataset_handler(self, mode):
        if mode == "train":
            return self.args.train_split
        elif mode == "val":
            return self.args.val_split
        elif mode == "test":
            return self.args.test_split
        else:
            raise ValueError(
                f"Mode {mode} is not known. Choose between train, val or test."
            )


if __name__ == "__main__":
    pass
