from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import os 
import numpy as np 
import pandas as pd 
import copy 

import torch
from torchvision import transforms 
from collections import defaultdict
import glob 
import cv2 

PATH_TO_IMG = r"data/"



class CarDataset(torch.utils.data.Dataset): 
    """
    Dataset Class
    """

    def __init__(self, split: str, transform: Optional[transforms.Compose]): 

        super().__init__()
        
        self.split = split 
        self.data = self.load_images()
        self.transform = transform 

    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, idx): 
        
        file = self.data.iloc[idx]
        img_path = os.path.join(PATH_TO_IMG, "imgs", file["filename"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hood_score = torch.as_tensor(file["perspective_score_hood"])
        backdoor_left_score = torch.as_tensor(file["perspective_score_backdoor_left"])

        if self.transform: 
            sample = self.transform(image=image)
            image = sample["image"]

        
        return image, hood_score.type(torch.float32).unsqueeze(0), backdoor_left_score.type(torch.float32).unsqueeze(0)
    
    
    def load_images(self): 
        return pd.read_pickle(os.path.join(PATH_TO_IMG,self.split))