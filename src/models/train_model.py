#import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import cv2
import PIL
import math
import matplotlib.image as mpimg
from pathlib import Path
import logging

#########################_import_own_functions_####################################
from PathManagement import PathManagement
from Config import Config
from EpsilonDataset import EpsilonDataset
from data_for_model import prepare_data, saveModel
from loss_function import customLossFunction, singleCustomLossFunction

######################################################################################################

pathManagement=PathManagement()
config=Config(pathManagement)

#########################################################################################################

##########################################################################################################

dataloaders = prepare_data(config)
dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val']}
dataset = EpsilonDataset(config.data_root_dir, config.dataset_metadata, transform=config.data_transforms)
train_features, train_labels=next(iter(dataloaders["train"]))

print("Device: ", config.device())
print("Dataloader train len: ", len(dataloaders["train"]), "val len: ", len(dataloaders["val"]))


########################################################################################################

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = -100000
    
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

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

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss+=loss

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            #print("epoch_loss: ", epoch_loss, "running_loss: ", running_loss, "dataset_sizes[phase]: ", dataset_sizes[phase])
            epoch_acc = 1-epoch_loss

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

############################################################################################################

model_ft = models.resnet18(pretrained=True)
#todo mozna sprobowac wiekszego resneta
#todo najpierw uczy sie siec zamrozona i na poczatku uczy sie tylko ostatnie 
#     warstwy i dopiero jak dobrze pojdzie to odmrazamy
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 1)

model_ft = model_ft.to(config.device())

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=config.learning_rate, momentum=config.momentum)


# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=config.step_size, gamma=config.gamma)

#####################################################################################################
model_ft = train_model(model=model_ft, criterion=customLossFunction, optimizer=optimizer_ft, scheduler=exp_lr_scheduler,
                       num_epochs=config.epochs)

#####################################################################################################

saveModel(model=model_ft, config=config)
###################################################################################################

#state_dict = torch.load(pathManagement.modelSavePath(dataPlace = config.data_place) + config.model_name_to_save) #to check, is everything ok
tempPathToLoad = pathManagement.modelSavePath(dataPlace = config.data_place) + config.model_name_to_save

state_dict = torch.load(tempPathToLoad, map_location=config.device())

del tempPathToLoad
#print(state_dict.keys())


#########################____TESTOWANIE___########################################################
######################____DO_POSPRZĄTANIA___######################################################
device="cpu"
#model_ft = models.resnet18(pretrained=True)
#num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, 1)
#
#model_ft = model_ft.to(device)
#model_ft.load_state_dict(torch.load('Conv_RealPhotos_0_9745.pht'))
tempPathToLoad = pathManagement.modelSavePath(dataPlace = config.data_place) + config.model_name_to_save #temporary path
model_ft2=torch.load(tempPathToLoad)
model_ft2.eval()
#dataloaders = prepare_data(config)
dataloaders = prepare_data(config, train=False)
device="cuda"
allDiffs=0.0
j=0
for images, labels in dataloaders['val']:
    images, labels = images.cuda(), labels.cuda()
    outputs=model_ft2(images)
    for i in range(len(outputs)):
        diff=abs(float(labels[i]-outputs[i]))
        allDiffs+=diff
        j+=1
        if (j%100==0): print(j, "mean:", allDiffs/j)
        #print("j:", j, "label: ", float(labels[i]), "output: ", float(outputs[i]), "diff=", diff)

print("mean", allDiffs/j)

#######################################################################################################################

device="cpu"
#model_ft = models.resnet18(pretrained=True)
#num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, 1)
#
#model_ft = model_ft.to(device)
#model_ft.load_state_dict(torch.load('Conv_RealPhotos_0_9745.pht'))

tempPathToLoad = pathManagement.modelSavePath(dataPlace = config.data_place) + config.model_name_to_save #temporary path
print(tempPathToLoad)
model_ft2=torch.load(tempPathToLoad)
del tempPathToLoad
model_ft2.eval()
dataloaders = prepare_data(config, train=False)
device="cuda"
allDiffs=0.0
j=0

########################################################################################################################
for images, labels in dataloaders['test']:
    images, labels = images.cuda(), labels.cuda()
    outputs=model_ft2(images)
    for i in range(len(outputs)):
        #diff=abs(float(labels[i]-outputs[i]))
        #diff=singleCustomLossFunction(outputs[i], labels[i])
        diff = float(min( abs(abs(float(labels[i])-abs(outputs[i]))) , abs(1-float((abs(labels[i])-abs(outputs[i]))))))
        allDiffs+=float(diff)
        j+=1
        if (j%100==0): print(j, "mean:", allDiffs/j)
        #print("j:", j, "label: ", float(labels[i]), "output: ", float(outputs[i]), "diff=", diff)

print("mean", allDiffs/j)

########################################################################################################################

##### with printing charts
class Stats:
    class Bin:
        def __init__(self):
            self.value=0
            self.count=0
        
    
    def __init__(self, binCount):
        self.binAmmount=binCount
        self.bin=self.Bin()
        self.bins=[]

        for i in range(self.binAmmount):
            internalBin=self.Bin()
            self.bins.append(internalBin)

    def CreateBins(self):
        for i in range(self.binAmmount):
            internalBin=self.Bin()
            self.bins.append(internalBin)

    def AddCalculation(self, epsilon, error):
        #print(len(self.bins), int(epsilon/(1/self.binAmmount)), epsilon, error)
        self.bins[int(epsilon/(1/self.binAmmount))].value+=error
        self.bins[int(epsilon/(1/self.binAmmount))].count+=1


device="cpu"

tempPathToLoad = pathManagement.modelSavePath(dataPlace = config.data_place) + config.model_name_to_save #temporary path

model_ft2=torch.load(tempPathToLoad)

del tempPathToLoad

model_ft2.eval()


config.batch_size=1
dataloaders = prepare_data(config, train=False)
device="cuda"
allDiffs=0.0
stats=Stats(100)
j=0

numberOfSamples = np.zeros([100,1],dtype=int) #.
minError = np.ones([100,1],dtype=float)#.
maxError = np.zeros([100,1],dtype=float) #.
tempBinNumber = 0 #.


for images, labels in dataloaders['test']:
    images, labels = images.cuda(), labels.cuda()
    outputs=model_ft2(images)
    for i in range(len(outputs)):
        diff=min(abs(1-abs(float(labels[i]-outputs[i]))) , abs(float(labels[i]-outputs[i])))
        label=copy.deepcopy(float(labels[i]))
        output=copy.deepcopy(float(outputs[i]))
        allDiffs+=diff
        j+=1
        stats.AddCalculation(label, diff)
        #print("label: ", label,"diff: ", diff) #.
        tempBinNumber = int(label*1000 // 10) #.
        numberOfSamples[tempBinNumber] = numberOfSamples[tempBinNumber] + 1 #.
        if minError[tempBinNumber] > diff:
            minError[tempBinNumber] = diff
        if maxError[tempBinNumber] < diff:
            maxError[tempBinNumber] = diff

        if (j%100==0): print(j, "mean:", allDiffs/j)
        #print("j:", j, "label: ", float(labels[i]), "output: ", float(outputs[i]), "diff=", diff)
       
        #plt.imshow(images)
        #plt.show
        #out1 = torchvision.utils.make_grid(inputs)
        #imshow(out1,"abc")

#print(numberOfSamples) #.
x=[]
y=[]

for i in range (len(stats.bins)):
    x.append(1/stats.binAmmount*i)
    if(stats.bins[i].count==0): y.append(0)
    else: 
        y.append(stats.bins[i].value/stats.bins[i].count)
        print(stats.bins[i].value/stats.bins[i].count)
plt.plot(x,y)
plt.title(config.model_name_to_save)
plt.xlabel("Epsilon")
plt.ylabel("EpsilonError")
plt.show()

plt.plot(x, numberOfSamples) #.
plt.ylabel('number of samples') #.
plt.show() #.

plt.plot(x, minError) #.
plt.ylabel('m in error') #.
plt.show() #.

plt.plot(x, maxError) #.
plt.ylabel('max error') #.
plt.show() #.
#print("mean", allDiffs/j)


##########################___FUTURE___#######################################
#todo zamrozic i odmrozic .
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

visualize_model(model_conv)

plt.ioff()
plt.show()