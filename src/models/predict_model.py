
import copy
import statistics
from matplotlib import pyplot as plt
import numpy as np
import torch
from model_functions.Statistics import Stats
from model_functions.Config import Config
from model_functions.PathManagement import PathManagement
from model_functions.data_for_model import prepare_data
from model_functions.loss_function import numpy_single_custom_loss_function
from captum.attr import IntegratedGradients

pathManagement=PathManagement(dataType="original",
                                noiseType="noised",
                                centerInTheMiddle=True,
                                purposeData="test")


config=Config(pathManagement)
config.setModelNameToRead("hardtanh_1_generated_mixed.pth")


###################################################################################################################

tempPathToLoad = pathManagement.getModelSavePath() + config.model_name_to_read #temporary path
print("model path: " + tempPathToLoad)
model_ft2=torch.load(tempPathToLoad)
del tempPathToLoad

model_ft2.eval()
dataloaders = prepare_data(config, train=False)
allDiffs=0.0
max_single_diff = 0.0
j=0



#############################################_charts etc_##############################################################
##### with printing charts

config.batch_size=1
dataloaders = prepare_data(config, train=False)
allDiffs=0.0
stats=Stats(binCount = 100)
j=0


for images, labels in dataloaders['test']:
    images, labels = images.cuda(), labels.cuda()
    outputs=model_ft2(images)
    outputs = torch.sum(outputs,1)/512
##########
    diff = numpy_single_custom_loss_function(output = outputs[0], label = labels[0])

    label=copy.deepcopy(float(labels[0]))
    output=copy.deepcopy(float(outputs[0]))
    allDiffs+=diff
    j+=1
    stats.AddCalculation(label, diff)
    stats.statistics.addReturnedStatistics(output=output, label = label, lossCalculatedError=diff)

    if (j%100==0): print(j, "mean:", allDiffs/j)



stats.statistics.plotStatistics()
stats.statistics.plotReturnedStatistics()




input("")