import os
import numpy as np
import pandas as pd
import torch
from tifffile import imread


class CellPaintingDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
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
    x = CellPaintingDataset("../../data/processed", "../../data/data.csv")
    print(x.__getitem__(0))