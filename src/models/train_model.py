import numpy as np
import matplotlib.pyplot as plt
import time
import copy
#########################___import_torch_fun___####################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import  models
from torch.utils.tensorboard import SummaryWriter #to print to tensorboard
#########################_import_own_functions_####################################
from model_functions.PathManagement import PathManagement
from model_functions.Config import Config
from model_functions.EpsilonDataset import EpsilonDataset
from model_functions.data_for_model import prepare_data, saveModel, saveEpochModel
from model_functions.loss_function import custom_loss_function


######################################################################################################

pathManagement=PathManagement(dataType="generated", noiseType="noised", centerInTheMiddle=False, purposeData="training")
config=Config(pathManagement)



##########################################################################################################

dataloaders = prepare_data(config, train = True, datasetType = "baseline")
dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val']}
train_features, train_labels=next(iter(dataloaders["train"]))

print(config.model_name_to_save)
print("Device: ", config.device())
print("Dataloader train batch quantity: ", len(dataloaders["train"]), "val batch quantity: ", len(dataloaders["val"]))


########################################################################################################

def train_model(model, criterion, optimizer, scheduler, num_epochs, model_name):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0

            temp_quantity = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(config.device())

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    #plt.imshow(inputs[0].cpu().numpy()[0], cmap='gray') #for plot in debug etc.
                    outputs = model(inputs) 
                    #outputs = torch.sum(outputs,1)/512 
                    loss = criterion(outputs, labels)
                    temp_quantity += 1
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss+=loss

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = 1-epoch_loss

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                str(phase), float(epoch_loss), float(epoch_acc)))
            
            writerTensorBoard.add_scalar('Loss/'+str(phase), float(epoch_loss), epoch)
            writerTensorBoard.add_scalar('Accuracy/'+str(phase), float(epoch_acc), epoch)

            if phase == 'train':
                saveEpochModel(model=model, config=config, epoch_nr=epoch + 1, model_name=model_name)
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
    torch.save(model,"model_temp")
    return model

#############################___LEARNING_PROCES___###############################################
withoutFreeze = True
freeze = False
 


if withoutFreeze == True:
    #modelName = "compare_freeze_without.pth" #temporary here
    #########################################################################################################
    writerTensorBoard = SummaryWriter(f'tensorBoard/tensorBoard_' + config.model_name_to_save) # declare tensorboard


    model_ft = models.resnet18(pretrained=True)
    # model_ft.fc = nn.Hardtanh(min_val=0.0, max_val=1.0)
    num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs, 512)
    classifier_layer = nn.Sequential(
            #nn.Linear(2048 , 512), #resnet50
            nn.Linear(512 , 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256 , 128),
            nn.Linear(128 , 2),
            nn.Linear(2 , 1),

            #nn.Hardtanh(min_val=0.0, max_val=1.0)
        )
    #model_ft = nn.Sequential(model_ft, classifier_layer)
    model_ft.fc = classifier_layer

    model_ft = model_ft.to(config.device())

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=config.learning_rate, momentum=config.momentum)


    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=config.step_size, gamma=config.gamma)

    #####################################################################################################
    model_ft = train_model(model=model_ft, criterion=custom_loss_function, optimizer=optimizer_ft, scheduler=exp_lr_scheduler,
                        num_epochs=config.epochs, model_name=config.model_name_to_save)

    saveModel(model=model_ft, config=config, model_name=config.model_name_to_save)
    ###################################################################################################

if freeze == True:
    modelName = "bathSize4_freeze.pth" #temporary here
    #########################################################################################################
    writerTensorBoard = SummaryWriter(f'tensorBoard/tensorBoard_' + modelName) # declare tensorboard

    model_conv = models.resnet18(pretrained=True)

    for param in model_conv.parameters():
        param.requires_grad = False


    num_ftrs = model_conv.fc.in_features
    #model_conv.fc = nn.Linear(num_ftrs, 1)
    #model_conv.fc = nn.Sequential(nn.Linear(num_ftrs, 1), torch.nn.Hardtanh(0.0,1.0))

    classifier_layer = nn.Sequential(
        #nn.Linear(2048 , 512), #resnet50
        nn.Linear(512 , 256),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256 , 128),
        nn.Linear(128 , 2),
        nn.Linear(2 , 1),

        nn.Hardtanh(min_val=0.0, max_val=1.0)
    )
    #model_ft = nn.Sequential(model_ft, classifier_layer)
    model_conv.fc = classifier_layer
    
    model_conv = model_conv.to(config.device())


    # Observe that all parameters are being optimized
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=config.learning_rate, momentum=config.momentum)


    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=config.step_size, gamma=config.gamma)
    

    #####################################################################################################
    model_conv = train_model(model=model_conv, criterion=custom_loss_function, optimizer=optimizer_conv, scheduler=exp_lr_scheduler,
                        num_epochs=config.epochs, model_name=modelName)

    saveModel(model=model_conv, config=config, model_name=modelName)