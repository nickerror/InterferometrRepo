import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
#import wandb
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import lr_scheduler
from tqdm.auto import tqdm

from CellPaintingDataset import CellPaintingDataset
#from CellPaintingModel import CNN
from __init__ import Config

device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(0)



def get_stratified_sampler(config, generator):
    """
    Parameters
    ----------
    config : TYPE
        DESCRIPTION.
    generator : TYPE
        DESCRIPTION.

    Returns
    -------
    train_sampler : TYPE
        train samplers.
    val_sampler : TYPE
        value.
    test_sampler : TYPE
        test samples.
    """
    
    df = pd.read_csv(config.dataset_metadata) #read data from CSV to var df
    cartridges_names = list(df.folder.unique())
    test_cartridge = cartridges_names.pop(np.random.randint(0, len(cartridges_names)))
    test_index = df[df.folder.str.match(test_cartridge)].index.to_numpy()
    train_df = df[~df.index.isin(test_index)]
    y = train_df[['compound_label']].to_numpy()
    x = np.array(list(range(len(y)))).reshape(-1, 1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    (train_index, val_index) = list(sss.split(x, y))[0]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_index, generator=generator)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_index, generator=generator)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_index, generator=generator)
    return train_sampler, val_sampler, test_sampler

def prepare_data(config, data_transform):
    #dataset --> object from class CellPaintingDataset
    dataset = CellPaintingDataset(config.data_root_dir, config.dataset_metadata, transform=data_transform)
    g = torch.Generator(device=device).manual_seed(0)
    train_sampler, val_sampler, test_sampler = get_stratified_sampler(config, g)
    loader_params = dict(dataset=dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                         pin_memory=config.pin_memory, generator=g)
    train_loader = torch.utils.data.DataLoader(**loader_params, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(**loader_params, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(**loader_params, sampler=test_sampler)
    return {'train': train_loader, 'val': validation_loader, 'test': test_loader}

#start program:
config = Config()
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[163.24, 536.39, 425.26, 581.64],
                          std=[ 204.27, 1386.95,  917.2 ,  519.7 ])
])

print(data_transform)
dataloaders = prepare_data(config, data_transform)
