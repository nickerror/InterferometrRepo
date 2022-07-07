
import copy
from matplotlib import pyplot as plt
import numpy as np
import torch
from model_functions.Statistics import Stats
from model_functions.Config import Config
from model_functions.PathManagement import PathManagement
from model_functions.data_for_model import prepare_data
from model_functions.loss_function import numpy_single_custom_loss_function, torch_single_custom_loss_function


pathManagement=PathManagement(dataType="generated", 
                                noiseType="noised", 
                                centerInTheMiddle=False, 
                                purposeData="test")

config=Config(pathManagement)
config.setModelNameToRead("3_generated_mixed.pth")


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

#device="cpu"

config.batch_size=1
dataloaders = prepare_data(config, train=False)
#device="cuda"
allDiffs=0.0
stats=Stats(binCount = 100)
j=0

numberOfSamples = np.zeros([100,1],dtype=int) #.
minError = np.ones([100,1],dtype=float)#.
maxError = np.zeros([100,1],dtype=float) #.
tempBinNumber = 0 #.


for images, labels in dataloaders['test']:
    images, labels = images.cuda(), labels.cuda()
    outputs=model_ft2(images)

##########
    diff = numpy_single_custom_loss_function(output = outputs[0], label = labels[0])

    label=copy.deepcopy(float(labels[0]))
    output=copy.deepcopy(float(outputs[0]))
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


stats.statistics.plotStatistics()

plt.subplot(2,2,1)       
plt.plot(x,y)
plt.title(config.model_name_to_read)
plt.xlabel("Epsilon")
plt.ylabel("EpsilonError")
plt.show()

#plt.clf()
plt.subplot(2,2,2)
plt.plot(x, numberOfSamples) #.
plt.ylabel('number of samples') #.
plt.show() #.

#plt.clf()
plt.subplot(2,2,3)
plt.plot(x, minError) #.
plt.ylabel('min error') #.
plt.show() #.

#plt.clf()
plt.subplot(2,2,4)
plt.plot(x, maxError) #.
plt.ylabel('max error') #.
plt.show() #.
#print("mean", allDiffs/j)

input("")