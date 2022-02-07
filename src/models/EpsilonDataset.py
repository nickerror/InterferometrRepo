import os
import numpy as np
import pandas as pd
import torch
from tifffile import imread


class EpsilonDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        #self.annotations = pd.read_csv(annotation_file,skiprows=1)
        self.annotations = pd.read_csv(annotation_file,skiprows=0)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img = imread(os.path.join(self.root_dir, self.annotations.iloc[index, 0])).astype(np.float32)
        y_label = self.annotations.loc[index, 'compound_label']
        if self.transform is not None:
            img = self.transform(img)
        return img, y_label


if __name__ == "__main__":
    x = EpsilonDataset("../../data/processed", "../../data/data.csv")
    print(x.__getitem__(0))
    
#Test
from __init__ import Config
import torchvision.transforms as transforms
config = Config()
data_transform = transforms.Compose([
    transforms.CenterCrop(488),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])
dataset = EpsilonDataset(config.data_root_dir, config.dataset_metadata, transform=data_transform)
