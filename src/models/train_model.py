from IPython.display import clear_output 
clear_output()

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os
import copy
import pandas as pd
import cv2
import PIL
import math

#for test - number of the photo loaded
photoLoadedNo = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Config:
    def __init__(self):
        self.epochs = 25
        self.cuda=True
        self.num_classes = 1
        self.batch_size = 4
        self.learning_rate = 0.01
        self.dataset = "InterferometerPhoto"
        #self.architecture = "CNN"
        self.pin_memory = True
        self.momentum = 0.9
        self.step_size = 3
        self.gamma = 0.1
        self.dataset_metadata = "../../data/raw/1channel/reference/epsilon.csv" # will change for processed
        self.num_workers = 0
        self.data_root_dir = "../../data/raw/1channel/photo" # will change for processed
        self.data_transforms = transforms.Compose([
                #transforms.CenterCrop(448),
                #transforms.Resize(224),#############################Lub od razu centercrop(224)
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.491, 0.491, 0.491],
                                      std=[0.210, 0.210, 0.210]) 
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

class EpsilonDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        #self.annotations = pd.read_csv(annotation_file,skiprows=1)
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

def prepare_data(config):
    #dataset --> Photos from interferometer
    dataset = EpsilonDataset(config.data_root_dir, config.dataset_metadata, transform=config.data_transforms)

    g = torch.Generator(device=device).manual_seed(0)
    datasetLen=dataset.__len__()
    trainLen=int(0.8*datasetLen)
    valLen=int(0.2*datasetLen)
    testLen=int(0.0*datasetLen)

    mean=0
    std=0
    trainSize=0.8
    valSize=0.2
    trainIndex, valIndex, testIndex=[],[],[]

    for i in range (datasetLen):
        randomNumber=np.random.rand()
        #print(randomNumber)
        if(randomNumber<trainSize): trainIndex.append(i)
        else: valIndex.append(i)
        img,epsilon=dataset[i]
        img=np.array(img)
        mean+=np.mean(img)
        std+=np.std(img)
         
    valIndex=trainIndex

    mean=mean/datasetLen#/255
    std=std/datasetLen#/255
    #print("mean, std:",mean, std)
    #trainIndex=valIndex
    #testIndex=valIndex
    loader_params = dict(dataset=dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                         pin_memory=config.pin_memory, generator=g, shuffle=False)
    train_loader = torch.utils.data.DataLoader(**loader_params, sampler=trainIndex )
    validation_loader = torch.utils.data.DataLoader(**loader_params, sampler=valIndex )
    test_loader = torch.utils.data.DataLoader(**loader_params, sampler=testIndex)
    return {'train': train_loader, 'val': validation_loader, 'test': test_loader}


##!!!!!!!!!!!!___LOAD_DATA______!!!!!!!!!!!
config=Config()


#image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                          data_transforms[x])
#                  for x in ['train', 'val']}
dataloaders = prepare_data(config)
dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val']}
#class_names = image_datasets['train'].classes


print("Device: ", device)
print("Dataloader train len: ", len(dataloaders["train"]), "val len: ", len(dataloaders["val"]))

##!!!!!!!!!!!!___Training the model______!!!!!!!!!!!
def customLossFunction(outputs, labels):
    totalLoss=0.0
    #print(len(outputs))
    for i in range (len(outputs)):
        loss=min(abs(labels[i]%1-outputs[i]%1), 1-abs(labels[i]%1-outputs[i]%1))
        totalLoss+=loss
        #totalLoss=(100*loss)*(100*loss)
    totalLoss/=len(outputs)
    #totalLoss=math.sqrt(totalLoss)
    return totalLoss

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #_, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss+=loss
                #running_loss += loss.item() * inputs.size(0)
                #running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = 1-epoch_loss#running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                str(phase), float(epoch_loss), float(epoch_acc)))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(float(best_acc)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

##!!!!!!!!!!!!___Finetuining the convent______!!!!!!!!!!!
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 1)

model_ft = model_ft.to(device)

#criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.003, momentum=0.9)


# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#print((dataloaders['train'])[0])


##!!!!!!!!!!!!___Execute train model______!!!!!!!!!!!
model_ft = train_model(model_ft, customLossFunction, optimizer_ft, exp_lr_scheduler, num_epochs=config.epochs)






