import os
import numpy as np
import pandas as pd
import torch
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#from tifffile import imread
import PIL
from __init__ import Config

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

# def tensor_to_image(tensor):
#     tensor = tensor*255
#     tensor = np.array(tensor, dtype=np.uint8)
#     if np.ndim(tensor)>3:
#         assert tensor.shape[0] == 1
#         tensor = tensor[0]
#     return PIL.Image.fromarray(tensor)


# if __name__ == "__main__":
#     config=Config()
#     data_transform = transforms.Compose([
#     transforms.CenterCrop(488),
#     transforms.Resize(224),
#     transforms.ToTensor()#,
#     #transforms.Normalize(mean=[0.485, 0.456, 0.406],
#     #                      std=[0.229, 0.224, 0.225]) #jak to dziala
#     ])
#     dataset = EpsilonDataset(config.data_root_dir, config.dataset_metadata, data_transform)
#     #print(np.array(cv2.imread(os.path.join(dataset.root_dir, str(str("%05d" %dataset.annotations.imgnr[0]))+ ".png"))))
#     #print(os.path.join(dataset.root_dir, str(str("%05d" %dataset.annotations.imgnr[0])+".png")))
#     #print(dataset.annotations.imgnr[1])
#     img,label=dataset[0]
#     print(label)
#     print(img.shape)
#     #img=tensor_to_image(img)
    
#     #img.show()
    
    

