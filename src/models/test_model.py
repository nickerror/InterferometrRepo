
from matplotlib import pyplot as plt
import torch
from model_functions.Config import Config
from model_functions.PathManagement import PathManagement
from model_functions.data_for_model import prepare_data


pathManagement=PathManagement()
config=Config(pathManagement)



###################################################################################################################

tempPathToLoad = pathManagement.modelSavePath(dataPlace = config.data_place) + config.model_name_to_read #temporary path
print(tempPathToLoad)
model_ft2=torch.load(tempPathToLoad)
del tempPathToLoad
model_ft2.eval()
dataloaders = prepare_data(config, train=False)
allDiffs=0.0
max_single_diff = 0.0
j=0


#############################_TEST_TIME_#################################################

for images, labels in dataloaders['test']:
    images, labels = images.cuda(), labels.cuda()
    outputs=model_ft2(images)
    for i in range(len(outputs)):
        #diff=singleCustomLossFunction(outputs[i], labels[i])
        diff = float(min( abs(abs(float(labels[i])-abs(outputs[i]))) , abs(1-float((abs(labels[i])-abs(outputs[i]))))))
        allDiffs+=float(diff)
        if diff > max_single_diff:
            max_single_diff = diff
            print("new max single diff: ", max_single_diff)
        j+=1
        if (j%100==0): print(j, "mean diff:", allDiffs/j)
        #print("j:", j, "label: ", float(labels[i]), "output: ", float(outputs[i]), "diff=", diff)

print("mean diff:", allDiffs/j)
print("mean diff in deg:", (allDiffs/j)*2*360)
print("max single diff:", max_single_diff)

if False:
    #############################################_charts etc_##############################################################
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
    plt.ylabel('min error') #.
    plt.show() #.

    plt.plot(x, maxError) #.
    plt.ylabel('max error') #.
    plt.show() #.
    #print("mean", allDiffs/j)