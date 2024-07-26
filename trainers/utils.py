
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig




def get_training_augmentations(cfg: DictConfig):
    if "vit" in cfg.arch: 
        train_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.ShiftScaleRotate(shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1), rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    elif cfg.encoder_weights == "imagenet": 
        train_transform = A.Compose(
        [
            A.LongestMaxSize(cfg.img_size),
            A.ShiftScaleRotate(shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1), rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    else: 
        train_transform = A.Compose(
        [
            A.LongestMaxSize(cfg.img_size),
            A.ShiftScaleRotate(shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1), rotate_limit=30, p=0.3),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            ToTensorV2(),
        ]
    )
    return A.Compose(train_transform)



def get_validation_augmentations(cfg: DictConfig): 
    if "vit" in cfg.arch: 
        val_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    elif cfg.encoder_weights == "imagenet": 
        val_transform = A.Compose(
        [
            A.LongestMaxSize(cfg.img_size, cfg.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    else: 
        val_transform = A.Compose(
        [
            A.LongestMaxSize(cfg.img_size, cfg.img_size), # keep width/height ratio 
            ToTensorV2(),
        ]
    )
    return A.Compose(val_transform)