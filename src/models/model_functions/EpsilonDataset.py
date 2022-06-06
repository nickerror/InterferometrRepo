import os
import numpy as np
import pandas as pd
import torch
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#from tifffile import imread
import PIL

class EpsilonDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file,skiprows=0, delim_whitespace=' ')
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img = np.array(cv2.imread(os.path.join(self.root_dir, str(str("%05d" %self.annotations.imgnr[index]))+ ".png"))).astype(np.float32)
        img=PIL.Image.fromarray(np.uint8(img))
        y_label = self.annotations.eps[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, y_label

    
    

