
import os 

import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger

import torch 
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter
import hydra 
from omegaconf import DictConfig
import builtins


from dataset import CarDataset
from trainers import LightingModel



@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None: 
    

    model = LightingModel(cfg)
    print(f"Model Number of Parameters: {model.num_params}")
    train_dataloader = model._create_dataloader(mode = "train")
    val_dataloader = model._create_dataloader(mode = "val")
    test_dataloader = model._create_dataloader(mode = "test")
    
    for img, score1, score2 in train_dataloader: 
        break

    """pred = model.forward(img)
    if cfg.joint_training: 
        pred1, pred2 = pred 
        print(img.shape, score1.shape, score2.shape, pred1.shape, pred2.shape )

    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(pred1, score1) + loss_fn(pred2, score2)
    print(score1.dtype)
    print(pred1.dtype)
    print(score2.dtype)
    print(pred2.dtype)
    print(loss.item())"""



    """
    print("TRAIN")
    print(img.shape, score1.shape, score2.shape)
    for img, score1, score2 in val_dataloader: 
        break
    print("VAL")
    print(img.shape, score1.shape, score2.shape)
    for img, score1, score2 in test_dataloader: 
        break
    print("TEST")
    print(img.shape, score1.shape, score2.shape)


    print(len(train_dataloader), len(val_dataloader), len(test_dataloader))"""


    
    logger = TensorBoardLogger(save_dir=os.getcwd(), name="lightning_logs/"+f"arch_{cfg.arch}_batchsize_{cfg.batch_size}_weights_{cfg.encoder_weights}_img_size_{cfg.img_size}_augmentations_{cfg.use_augmentations}_joint_training_{cfg.joint_training}_hood_training_{cfg.hood_training}_activation_{cfg.activation}")
    
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

    

if __name__ == '__main__': 
    main()