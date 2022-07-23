
import copy
import multiprocessing
from cv2 import accumulate
from matplotlib import pyplot as plt
import torch
from model_functions.Statistics import Stats
from model_functions.Config import Config
from model_functions.PathManagement import PathManagement
from model_functions.data_for_model import prepare_data
from model_functions.loss_function import numpy_single_custom_loss_function
from model_functions.Visualisation import CaptumVisualisation

pathManagement=PathManagement(dataType="generated",
                                noiseType="noised",
                                centerInTheMiddle=False,
                                purposeData="test")


config=Config(pathManagement)
config.setModelNameToRead("hardtanh_1_generated_mixed.pth")
config.setBathSize(1)

###################################################################################################################

tempPathToLoadModel = pathManagement.getModelSavePath() + config.model_name_to_read #temporary path
print("model path: " + tempPathToLoadModel)
model_ft=torch.load(tempPathToLoadModel)
del tempPathToLoadModel

model_ft.eval()

captumVisualisation = CaptumVisualisation(model_ft)

#############################################_charts etc_##############################################################
##### with printing charts

dataloaders = prepare_data(config, train=False)
accumulatedErrors=0.0
stats=Stats(binCount = 100)
numberOfChecked=0


for images, labels in dataloaders['test']:
    images, labels = images.cuda(), labels.cuda()
    outputs=model_ft(images)
    outputs_elements = torch.numel(outputs)
    outputs = torch.sum(outputs,1)/outputs_elements
    
    
    captumVisualisation.showCaptumVisualisation(images, visualize = False) #visualizes only if visualization has been turned on (parameter)
    
##########
    singleError = numpy_single_custom_loss_function(output = outputs[0], label = labels[0])

    label=copy.deepcopy(float(labels[0]))
    output=copy.deepcopy(float(outputs[0]))
    accumulatedErrors+=singleError
    numberOfChecked+=1
    stats.AddCalculation(label, singleError)
    stats.statistics.addReturnedStatistics(output=output, label = label, lossCalculatedError=singleError)

    if (numberOfChecked%100==0): print(numberOfChecked, "mean:", accumulatedErrors/numberOfChecked)



stats.statistics.plotStatistics()
stats.statistics.plotReturnedStatistics()


input("")