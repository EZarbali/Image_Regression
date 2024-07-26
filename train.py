import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig
from trainers import LightingModel


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model = LightingModel(cfg)
    print(f"Model Number of Parameters: {model.num_params}")
    train_dataloader = model._create_dataloader(mode="train")
    val_dataloader = model._create_dataloader(mode="val")

    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name="lightning_logs/"
        + f"arch_{cfg.arch}_batchsize_{cfg.batch_size}_weights_{cfg.encoder_weights}_img_size_{cfg.img_size}_augmentations_{cfg.use_augmentations}_joint_training_{cfg.joint_training}_hood_training_{cfg.hood_training}_activation_{cfg.activation}",
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="avg_val_loss")

    trainer = pl.Trainer(
        devices=1,
        accelerator="mps",
        precision=32,
        callbacks=[checkpoint_callback],
        logger=logger,
        max_epochs=cfg.epochs,
        strategy="auto",
        log_every_n_steps=1,
    )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
