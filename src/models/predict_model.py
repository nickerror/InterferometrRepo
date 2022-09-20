import copy
from tkinter import W
from matplotlib import pyplot as plt
import torch
from model_functions.Statistics import Stats
from model_functions.Config import Config
from model_functions.PathManagement import PathManagement
from model_functions.data_for_model import prepare_data
from model_functions.loss_function import numpy_single_custom_loss_function
from model_functions.Visualisation import CaptumVisualisation
# import pandas as pd
import csv
#data to set
pathManagement=PathManagement(dataType="original",
                                noiseType="noised",
                                centerInTheMiddle=False,
                                purposeData="test")

config=Config(pathManagement)
config.setModelNameToRead("loss_cos2.pth")
config.setBathSize(1)
visualizeCaptumImage = False
writeToCSV = False
chartStatsName = "model ResNet18 nauczony na danych generowanych, mieszanych"
#end of data to be set

##########################################_PARAMETRIZE_############################################################

tempPathToLoadModel = pathManagement.getModelSavePath() + config.model_name_to_read #temporary path
print("model path: " + tempPathToLoadModel)
model_ft=torch.load(tempPathToLoadModel)
del tempPathToLoadModel

model_ft.eval()

captumVisualisation = CaptumVisualisation(model_ft)

dataloaders = prepare_data(config, train=False)
accumulatedErrors=0.0
stats=Stats(chartName = chartStatsName, binCount = 100)
numberOfChecked=0

#write to CSV part
if writeToCSV:
    fileWithData = open('csvFiles/' + config.model_name_to_read + '.csv','w')
    writer = csv.writer(fileWithData)
    dataToSave = ['label', 'prediction']
    writer.writerow(dataToSave)
#end write CSV part


#############################################_PREDICTION_##############################################################
for images, labels in dataloaders['test']:
    images, labels = images.cuda(), labels.cuda()
    outputs = model_ft(images) #ToDo: write class "predict", with all transformations.
    outputs_elements = torch.numel(outputs)
    outputs = torch.sum(outputs,1)/outputs_elements
    singleError = numpy_single_custom_loss_function(output = outputs[0], label = labels[0])
    
    if writeToCSV:
        dataToSave = [labels[0].cpu().detach().numpy(), outputs[0].cpu().detach().numpy()]
        writer.writerow(dataToSave)
 ########################################_CHARTS_AND_INFORMATIONS_##############################################################   
    captumVisualisation.showCaptumVisualisation(images, label = labels[0], prediction = outputs[0], error = singleError, visualize = visualizeCaptumImage, showOriginalImage=True) 



    label=copy.deepcopy(float(labels[0]))
    output=copy.deepcopy(float(outputs[0]))
    accumulatedErrors+=singleError
    numberOfChecked+=1
    stats.AddCalculation(label, singleError)
    stats.statistics.addReturnedStatistics(output=output, label = label, lossCalculatedError=singleError)

    if (numberOfChecked%100==0): print(numberOfChecked, "mean:", accumulatedErrors/numberOfChecked)

if writeToCSV:
    fileWithData.close()
stats.statistics.plotStatistics()
stats.statistics.plotReturnedStatistics()


input("")