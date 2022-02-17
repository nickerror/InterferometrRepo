import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import wandb
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import lr_scheduler
from tqdm.auto import tqdm

from EpsilonDataset import EpsilonDataset
import torch.nn as nn
#from EpsilonModel import CNN
from __init__ import Config

device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"
print(device)
np.random.seed(0)




class CNN(nn.Module):
    def __init__(self, model, num_classes=9, train_net=False):
        super(CNN, self).__init__()
        self.model = model
        #if not train_net:
        #    for param in self.model.parameters():
        #        param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.device=device
        #layer = self.model.conv1
        #new_in_channels = 4
        #new_layer = nn.Conv2d(in_channels=new_in_channels,
        #                      out_channels=layer.out_channels,
        #                      kernel_size=layer.kernel_size,
        #                      stride=layer.stride,
        #                      padding=layer.padding,
        #                      bias=layer.bias)
        #copy_weights = 0
        #new_layers_weight = new_layer.weight.clone()
        #new_layers_weight[:, :layer.in_channels, :, :] = layer.weight.clone()
        #for i in range(new_in_channels - layer.in_channels):
        #    channel = layer.in_channels + i
        #    new_layers_weight[:, channel:channel + 1, :, :] = layer.weight[:, copy_weights:copy_weights + 1, ::].clone()
        #new_layer.weight = nn.Parameter(new_layers_weight)
        #self.model.conv1 = new_layer

    def forward(self, images):
        features = self.model(images)
        return features

def prepare_data(config, data_transform):
    #dataset --> Photos from interferometer
    dataset = EpsilonDataset(config.data_root_dir, config.dataset_metadata, transform=data_transform)

    g = torch.Generator(device=device).manual_seed(0)
    datasetLen=dataset.__len__()
    trainLen=int(0.6*datasetLen)
    valLen=int(0.1*datasetLen)
    testLen=int(0.3*datasetLen)

    trainIndex, valIndex, testIndex=[],[],[]

    for i in range (datasetLen):
        if(i<trainLen): trainIndex.append(i)
        elif (i<trainLen+valLen): valIndex.append(i)
        elif(i<trainLen+valLen+testLen): testIndex.append(i)
    trainIndex=valIndex
    testIndex=valIndex
    loader_params = dict(dataset=dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                         pin_memory=config.pin_memory, generator=g, shuffle=False)
    train_loader = torch.utils.data.DataLoader(**loader_params, sampler=trainIndex )
    validation_loader = torch.utils.data.DataLoader(**loader_params, sampler=valIndex )
    test_loader = torch.utils.data.DataLoader(**loader_params, sampler=testIndex)
    return {'train': train_loader, 'val': validation_loader, 'test': test_loader}

def prepare_model(model_architecture, config):
    model = CNN(model_architecture, num_classes=config.num_classes).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    return model, criterion, optimizer, exp_lr_scheduler


def train_log(loss, accuracy, example_ct, epoch, is_train):
    if is_train:
        #wandb.log({"epoch": epoch, "loss": loss, "acc": accuracy * 100.}, step=example_ct)
        print("epoch", epoch, "loss", loss, "acc", accuracy * 100.)
    else:
        #wandb.log({"epoch": epoch, "val_loss": loss, "val_acc": accuracy * 100.}, step=example_ct)
        print("epoch", epoch, "val_loss", loss, "val_acc", accuracy * 100.)


def train(model, criterion, optimizer, scheduler, dataloaders, config):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc, example_ct, batch_ct = 0.0, 0, 0
    for epoch in tqdm(range(config.epochs)):
        for phase in ['train', 'val']:
            model.train(phase == 'train')
            running_loss = 0.0
            running_corrects = 0
            num_examples = 0
            tempLoss=0
            with tqdm(dataloaders[phase], unit="batch", leave=False ) as tepoch:
                for _, (inputs, labels) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    labels = labels.to(device)
                    inputs = inputs.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        #_, predictions = torch.max(outputs, 1)
                        #print("\n", outputs.shape, labels.shape)
                        #print(outputs, labels)

                        #loss = criterion(outputs, labels)
                        loss=0
                        for i in range(len(outputs)):
                            #print(outputs[i], labels[i])
                            loss=loss+abs(labels[i]-outputs[i])
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    example_ct += len(inputs)
                    batch_ct += 1
                    #correct = torch.sum(predictions == labels.data).detach().cpu().numpy()
                    #accuracy = correct / config.batch_size
                    accuracy=1.0/loss
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += accuracy
                    num_examples += inputs.shape[0]
                    tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                    train_log(loss, accuracy, example_ct, epoch, phase == 'train')
            if phase == 'train':
                scheduler.step()
            #epoch_acc = running_corrects / num_examples
            epoch_acc=running_corrects/num_examples
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    #start program:
    config = Config()
    data_transform = transforms.Compose([
        transforms.CenterCrop(488),
        transforms.Resize(224),
        transforms.ToTensor()#,
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]) #jak to dziala
    ])
    print("data_transform", data_transform)
   
    dataloaders = prepare_data(config, data_transform)
    
    #for key in dataloaders.keys:
    #    print(key)
    print("list(dataloaders)", list(dataloaders))

    model = models.resnet18(pretrained=True)
    model, criterion, optimizer, scheduler = prepare_model(model, config)
    # with wandb.init(project="CellPainting", config=config.__dict__):
    model = train(model, criterion, optimizer, scheduler, dataloaders, config)


