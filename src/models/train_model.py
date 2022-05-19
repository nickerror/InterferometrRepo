# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

print('something more')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

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

###############################################################################################
class PathManagement:
    def __init__(self):
        self.__cloud_path_prefix = "/content/drive/My Drive/"
        #########___Data PATH___##############
        # FOR LOCAL DATA:
        # --REAL DATA:
        self._localDataset_metadata = "../data/raw/1channel/reference/epsilon.csv"
        self._localData_root_dir = "../data/raw/1channel/photo/"
        # --SELF-GENERATED DATA:
        # ----UNNOISED
        self._localData_metadata_generated_unnoised = "../data/generated/unnoised/reference/epsilon.csv"
        self._localData_root_dir_generated_unnoised = "../data/generated/unnoised/photo/"
        # ----NOISED
        self._localData_metadata_generated_noised = "../data/generated/unnoised/reference/epsilon.csv"
        self._localData_root_dir_generated_noised = "../data/generated/unnoised/photo/"

        # ON DRIVE:
        # --REAL DATA:
        self._cloudDataset_metadata = self.__cloud_path_prefix + "data/reference/real/epsilon_short.csv"
        self._cloudData_root_dir = self.__cloud_path_prefix + "data/photo/real/"
        # --SELF-GENERATED DATA:
        # ----UNNOISED
        self._cloudData_metadata_generated_unnoised = self.__cloud_path_prefix + "data/reference/generated/unnoised/epsilon.csv"
        self._cloudData_root_dir_generated_unnoised = self.__cloud_path_prefix + "data/photo/generated/unnoised/"
        # ----NOISED
        self._cloudData_metadata_generated_noised = self.__cloud_path_prefix + "data/reference/generated/noised/epsilon.csv"
        self._cloudData_root_dir_generated_noised = self.__cloud_path_prefix + "data/photo/generated/noised/"

        #########___Model PATH___##############
        self.__path_save_model_cloud = self.__cloud_path_prefix + "data/models/"
        self.__path_save_model_local = "../models/"

    def dataPath(self, dataPlace="local", dataType="original", isNoise=True):
        """! define correct data path using parameters

        @param dataPlace  data place can be 'local' or 'cloud'.
        @param dataType   data type can be 'original' or 'generated'.
        @param isNoise    only used in case of generated dataType.

        @return 2 path --> 1. with methadata, 2. with photo
        """
        if dataPlace == 'local':
            if dataType == 'original':
                return self._localDataset_metadata, self._localData_root_dir
            elif dataType == 'generated':
                if isNoise == False:
                    return self._localData_metadata_generated_unnoised, self._localData_root_dir_generated_unnoised
                else:
                    return self._localData_metadata_generated_noised, self._localData_root_dir_generated_noised
            else:
                return False
        elif dataPlace == 'cloud':
            if dataType == 'original':
                return self._cloudDataset_metadata, self._cloudData_root_dir
            elif dataType == 'generated':
                if isNoise == False:
                    return self._cloudData_metadata_generated_unnoised, self._cloudData_root_dir_generated_unnoised
                else:
                    return self._cloudData_metadata_generated_noised, self._cloudData_root_dir_generated_noised
            else:
                return False
        else:
            return False

    def modelSavePath(self, dataPlace="local"):
        """! define model save path depending on the save location

        @param dataPlace  data place can be 'local' or 'cloud'.

        @return model save path
        """
        if dataPlace == "local":
            return self.__path_save_model_local
        elif dataPlace == "cloud":
            return self.__path_save_model_cloud
        else:
            return False

########################################################################################################

pathManagement=PathManagement()

########################################################################################################

class Config:
    def __init__(self):
        # Variables to edit
        self.epochs = 20  # number of epochs
        self.num_classes = 1  # num classes in dataset
        # todo zwiekszyc batch_size -> 16 -> 32
        self.batch_size = 4  # batch size used for training (e.g. bath_size photos in one process)
        # todo sprawdzic Adama
        self.learning_rate = 0.001  # for SGD = 0.01, for Adam = 10^-4 -- 10^-3
        self.train_size = 0.8
        self.dataset = "InterferometerPhoto"
        # self.architecture = "CNN"
        self.pin_memory = True
        self.momentum = 0.9  # do Adama
        self.step_size = 7
        self.gamma = 0.1
        self.num_workers = 0
        self.model_name_to_save = "model_no_1.pth"
        self.data_place = "cloud"  # ="local"
        self.data_transforms = transforms.Compose([
            transforms.CenterCrop(448),
            transforms.Resize(224),  # first way is crop and later resize. Second way is CenterCrop right away.
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.491, 0.491],
                                 std=[0.210, 0.210, 0.210])])
        self._cuda = True  # GPU = True, CPU = False

        # variables not to edit here. You Can edit path in PathManagement Class.
        self.dataset_metadata, self.data_root_dir = pathManagement.dataPath(dataPlace=self.data_place,
                                                                            dataType="original",
                                                                            isNoise=True)

        # additional
        self.debug_mode = False

    def device(self):
        if self._cuda == True:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            return "cpu"

######################################################################################################

config=Config()

######################################################################################################

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

############################################################################################################

def prepare_data(config):
    # create time logger:
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                        level=50)  # 50 - critical, 40 - error, 30 - warning, 20 - info, 10 - debug, 0 - notset
    logging.debug('1. Start prepare_data')

    dataset = EpsilonDataset(config.data_root_dir, config.dataset_metadata, transform=config.data_transforms)

    g = torch.Generator(device=device).manual_seed(23)
    train_size = int(config.train_size * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=g)

    print("len(train_dataset):", len(train_dataset), "len(val_dataset):", len(val_dataset))

    loader_params = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                         pin_memory=config.pin_memory, generator=g, shuffle=True)

    train_loader = torch.utils.data.DataLoader(**loader_params, dataset=train_dataset)
    validation_loader = torch.utils.data.DataLoader(**loader_params, dataset=val_dataset)

    return {'train': train_loader, 'val': validation_loader}

#########################################################################################################

#import data from drive
from google.colab import drive
drive.mount('/content/drive')

#test connection:
img = mpimg.imread(config.data_root_dir + '03400.png') #test display img')
imgplot = plt.imshow(img)
plt.show()

##########################################################################################################

device = "cpu" #first calculations will be on CPU

dataloaders = prepare_data(config)
dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val']}

dataset = EpsilonDataset(config.data_root_dir, config.dataset_metadata, transform=config.data_transforms)

train_features, train_labels=next(iter(dataloaders["train"]))

print("Device: ", device)
print("Dataloader train len: ", len(dataloaders["train"]), "val len: ", len(dataloaders["val"]))

########################################################################################################

device = config.device() #another calculations with default
print("Device: ", device)

########################################################################################################

def customLossFunctionDebug(outputs, labels, totalLoss):
    print("NextOne")
    for i in range(len(outputs)):
        print("i: ", i, "label: ", float(labels[i]), "output:", float(outputs[i]), "diff= ",
              float(min(abs(abs(labels[i]) - abs(outputs[i])), abs(1 - (abs(labels[i]) - abs(outputs[i]))))))

    print("totalLoss:", float(totalLoss))
    return totalLoss


def customLossFunction(outputs, labels):
    totalLoss = 0.0
    for i in range(len(outputs)):
        # oneOutputLoss= abs(abs(labels[i])-(outputs[i]))
        # oneOutputLoss=min( abs(abs(labels[i])-abs(outputs[i])) , abs(1-(abs(labels[i])-abs(outputs[i]))))
        oneOutputLoss = torch.min(torch.abs(torch.abs(labels[i]) - torch.abs(outputs[i])),
                                  torch.abs(1 - (torch.abs(labels[i]) - torch.abs(outputs[i]))))
        totalLoss += oneOutputLoss
    totalLoss /= len(outputs)
    # customLossFunctionDebug(outputs=outputs, labels=labels, totalLoss=totalLoss)
    return totalLoss


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
                model.eval()  # Set model to evaluate mode

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
                running_loss += loss

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # print("epoch_loss: ", epoch_loss, "running_loss: ", running_loss, "dataset_sizes[phase]: ", dataset_sizes[phase])
            epoch_acc = 1 - epoch_loss

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

device = config.device()
model_ft = models.resnet18(pretrained=True)
#todo mozna sprobowac wiekszego resneta
#todo najpierw uczy sie siec zamrozona i na poczatku uczy sie tylko ostatnie
#     warstwy i dopiero jak dobrze pojdzie to odmrazamy
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 1)

model_ft = model_ft.to(device)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=config.learning_rate, momentum=config.momentum)


# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=config.step_size, gamma=config.gamma)
#print((dataloaders['train'])[0])

#####################################################################################################

model_ft = train_model(model=model_ft, criterion=customLossFunction, optimizer=optimizer_ft, scheduler=exp_lr_scheduler,
                       num_epochs=config.epochs)

#####################################################################################################

def saveModel(model, modelName):
    """! function to save model

    @param model        model to save
    @param modelName    name of model - prefered pith *.pth
    """
    tempPathToSave = pathManagement.modelSavePath(dataPlace=config.data_place) + modelName  # path to save

    torch.save(model_ft, tempPathToSave)
    # torch.save(model.state_dict(), tempPathToSave)

    print("model saved: " + config.data_place)

##################################################################################################

modelName = "3_model_0_9739.pth"

saveModel(model=model_ft, modelName = config.model_name_to_save)

###################################################################################################

#state_dict = torch.load(pathManagement.modelSavePath(dataPlace = config.data_place) + config.model_name_to_save) #to check, is everything ok
tempPathToLoad = pathManagement.modelSavePath(dataPlace = config.data_place) + config.model_name_to_save

state_dict = torch.load(tempPathToLoad, map_location=device)

del tempPathToLoad
#print(state_dict.keys())

###################################################################################################

device="cpu"
#model_ft = models.resnet18(pretrained=True)
#num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, 1)
#
#model_ft = model_ft.to(device)
#model_ft.load_state_dict(torch.load('Conv_RealPhotos_0_9745.pht'))
tempPathToLoad = pathManagement.modelSavePath(dataPlace = config.data_place) + config.model_name_to_save #temporary path
model_ft2=torch.load(tempPathToLoad)
del tempPathToLoad
model_ft2.eval()
dataloaders = prepare_data(config)
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

##################################################################################################

##### with printing charts
class Stats:
    class Bin:
        def __init__(self):
            self.value = 0
            self.count = 0

    def __init__(self, binCount):
        self.binAmmount = binCount
        self.bin = self.Bin()
        self.bins = []

        for i in range(self.binAmmount):
            internalBin = self.Bin()
            self.bins.append(internalBin)

    def CreateBins(self):
        for i in range(self.binAmmount):
            internalBin = self.Bin()
            self.bins.append(internalBin)

    def AddCalculation(self, epsilon, error):
        # print(len(self.bins), int(epsilon/(1/self.binAmmount)), epsilon, error)
        self.bins[int(epsilon / (1 / self.binAmmount))].value += error
        self.bins[int(epsilon / (1 / self.binAmmount))].count += 1


device = "cpu"

tempPathToLoad = pathManagement.modelSavePath(dataPlace=config.data_place) + config.model_name_to_save  # temporary path

model_ft2 = torch.load(tempPathToLoad)

del tempPathToLoad

model_ft2.eval()

config.batch_size = 1
dataloaders = prepare_data(config)
device = "cuda"
allDiffs = 0.0
stats = Stats(100)
j = 0

numberOfSamples = np.zeros([100, 1], dtype=int)  # .
minError = np.ones([100, 1], dtype=float)  # .
maxError = np.zeros([100, 1], dtype=float)  # .
tempBinNumber = 0  # .

for images, labels in dataloaders['val']:
    images, labels = images.cuda(), labels.cuda()
    outputs = model_ft2(images)
    for i in range(len(outputs)):
        diff = min(abs(1 - abs(float(labels[i] - outputs[i]))), abs(float(labels[i] - outputs[i])))
        label = copy.deepcopy(float(labels[i]))
        output = copy.deepcopy(float(outputs[i]))
        allDiffs += diff
        j += 1
        stats.AddCalculation(label, diff)
        # print("label: ", label,"diff: ", diff) #.
        tempBinNumber = int(label * 1000 // 10)  # .
        numberOfSamples[tempBinNumber] = numberOfSamples[tempBinNumber] + 1  # .
        if minError[tempBinNumber] > diff:
            minError[tempBinNumber] = diff
        if maxError[tempBinNumber] < diff:
            maxError[tempBinNumber] = diff

        if (j % 100 == 0): print(j, "mean:", allDiffs / j)
        # print("j:", j, "label: ", float(labels[i]), "output: ", float(outputs[i]), "diff=", diff)

        # plt.imshow(images)
        # plt.show
        # out1 = torchvision.utils.make_grid(inputs)
        # imshow(out1,"abc")

# print(numberOfSamples) #.
x = []
y = []

for i in range(len(stats.bins)):
    x.append(1 / stats.binAmmount * i)
    if (stats.bins[i].count == 0):
        y.append(0)
    else:
        y.append(stats.bins[i].value / stats.bins[i].count)
        print(stats.bins[i].value / stats.bins[i].count)
plt.plot(x, y)
plt.title(modelName)
plt.xlabel("Epsilon")
plt.ylabel("EpsilonError")
plt.show()

plt.plot(x, numberOfSamples)  # .
plt.ylabel('number of samples')  # .
plt.show()  # .

plt.plot(x, minError)  # .
plt.ylabel('min error')  # .
plt.show()  # .

plt.plot(x, maxError)  # .
plt.ylabel('max error')  # .
plt.show()  # .
# print("mean", allDiffs/j)