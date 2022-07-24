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

pathManagement=PathManagement(dataType="generated", noiseType="mixed", centerInTheMiddle=False, purposeData="training")
config=Config(pathManagement)

#########################################################################################################
writerTensorBoard = SummaryWriter(f'tensorBoard/tensorBoard_'+config.model_name_to_save) # declare tensorboard

##########################################################################################################

dataloaders = prepare_data(config)
dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val']}
train_features, train_labels=next(iter(dataloaders["train"]))

print(config.model_name_to_save)
print("Device: ", config.device())
print("Dataloader train batch quantity: ", len(dataloaders["train"]), "val batch quantity: ", len(dataloaders["val"]))


########################################################################################################

def train_model(model, criterion, optimizer, scheduler, num_epochs):
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
                    outputs = torch.sum(outputs,1)/512 
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

            saveEpochModel(model=model, config=config, epoch_nr=epoch + 1)
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

############################################################################################################

model_ft = models.resnet18(pretrained=True)
#todo mozna sprobowac wiekszego resneta
#todo najpierw uczy sie siec zamrozona i na poczatku uczy sie tylko ostatnie 
#     warstwy i dopiero jak dobrze pojdzie to odmrazamy

model_ft.fc = nn.Hardtanh(min_val=0.0, max_val=1.0)


model_ft = model_ft.to(config.device())

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=config.learning_rate, momentum=config.momentum)


# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=config.step_size, gamma=config.gamma)

#####################################################################################################
model_ft = train_model(model=model_ft, criterion=custom_loss_function, optimizer=optimizer_ft, scheduler=exp_lr_scheduler,
                       num_epochs=config.epochs)

saveModel(model=model_ft, config=config)
###################################################################################################

