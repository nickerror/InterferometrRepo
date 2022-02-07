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

from EpsilonDataset import EpsilonDataset
#from EpsilonModel import CNN
from __init__ import Config

device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(0)



def get_stratified_sampler(config, generator):    
    df = pd.read_csv(config.dataset_metadata) #read data from CSV to var "Data Frame"
    
    #cartridges_names = list(df.folder.unique()) #????
    cartridges_names = list(df.folds.unique()) #tworzy liste unikalnych wartosci
    
    test_cartridge = cartridges_names.pop(np.random.randint(0, len(cartridges_names))) #losuje jedna unikalna wartosc
    
    #test_index = df[df.folder.str.match(test_cartridge)].index.to_numpy()
    test_index = df[df.folds.str.match(test_cartridge)].index.to_numpy() #wyciaga wszystkie indeksy dla wartosci wyznaczonej wczesniej
    
    train_df = df[~df.index.isin(test_index)] #wywala z tablicy wszystkie wartosci okreslone jako testowe
    #y = train_df[['compound_label']].to_numpy()
    y = train_df[['epsilon']].to_numpy().astype(float) #zbior danych do uczenia
    x = np.array(list(range(len(y)))).reshape(-1, 1) #tworzy tablice o wielkosci zbioru uczacego
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0) #do podzialu. pozniej mozna ustawic wiecej splitow
    (train_index, val_index) = list(sss.split(x, y))[0]
    # for train_index, val_index in sss.split(x, y):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     x_train, x_val = x[train_index], x[val_index]
    #     y_train, y_val = y[train_index], y[val_index]
    #generator = torch.Generator(device=device).manual_seed(0)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_index, generator=generator)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_index, generator=generator)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_index, generator=generator)
    return train_sampler, val_sampler, test_sampler

def prepare_data(config, data_transform):
    #dataset --> object from class CellPaintingDataset
    dataset = EpsilonDataset(config.data_root_dir, config.dataset_metadata, transform=data_transform)
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
    transforms.CenterCrop(488),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]) #jak to dziala
])

print(data_transform)
#for test
dataset = EpsilonDataset(config.data_root_dir, config.dataset_metadata, transform=data_transform)
first_data = dataset[2]
features, labels = first_data
#end test

dataloaders = prepare_data(config, data_transform)
