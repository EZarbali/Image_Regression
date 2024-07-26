#!/bin/bash 



#echo "Training of from Scratch model starts" 
#python train.py finetune=True joint_training=True use_augmentations=False img_size=224 num_workers=1 arch=cnn activation=sigmoid

#echo "Training of resnet18 model starts" 
#python train.py finetune=True joint_training=True use_augmentations=False img_size=674 num_workers=1 arch=resnet18 activation=sigmoid

#echo "Training of resnet34 model starts" 
#python train.py finetune=True joint_training=True use_augmentations=False img_size=674 num_workers=1 arch=resnet34 activation=sigmoid

#echo "Training of vit model starts" 
#python train.py finetune=True joint_training=True use_augmentations=False img_size=224 num_workers=1 arch=vit activation=sigmoid epochs=8

#python train.py finetune=True joint_training=True use_augmentations=False img_size=224 num_workers=1 arch=vit activation=linear epochs=8

#python train.py finetune=True joint_training=True use_augmentations=False img_size=224 num_workers=1 arch=vit activation=relu epochs=8

#python train.py finetune=True joint_training=True use_augmentations=True img_size=224 num_workers=1 arch=vit activation=sigmoid epochs=16

#python train.py finetune=True joint_training=True use_augmentations=True img_size=224 num_workers=1 arch=vit activation=sigmoid epochs=16

python train.py finetune=True joint_training=True use_augmentations=False img_size=224 num_workers=1 arch=vit activation=sigmoid epochs=16

#python train.py finetune=True joint_training=False hood_training=False use_augmentations=False img_size=224 num_workers=1 arch=vit activation=sigmoid epochs=8

#python train.py finetune=True joint_training=False hood_training=True use_augmentations=False img_size=224 num_workers=1 arch=vit activation=sigmoid epochs=8
