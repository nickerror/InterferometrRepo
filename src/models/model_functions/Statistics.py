import numpy as np
from matplotlib import pyplot as plt
class Stats:
    class Statistics:
        def __init__(self, binCount = 100):
            self.__binCount = binCount
            self.__minError = 1.0
            self.__maxError = 0.0
            self.__binMinError = np.ones([binCount,1],dtype=float)
            self.__binMaxError = np.zeros([binCount,1],dtype=float) 
            self.__binMeanError = np.zeros([binCount,1], dtype=float)
            self.__samplesQuantityBin = np.zeros([binCount,1], dtype=float)

        
        def addErrorsToStatistic(self, epsilon, error, currentBinNumber = 0):
            if self.__maxError < error:
                self.__maxError = error
            if self.__minError > error:
                self.__minError = error

            if self.__binMaxError[currentBinNumber]  < error:
                self.__binMaxError[currentBinNumber] = error
            if self.__binMinError[currentBinNumber] > error:
                self.__binMinError[currentBinNumber] = error

            self.__binMeanError[currentBinNumber] += error
            self.__samplesQuantityBin[currentBinNumber] +=1

        def plotStatistics(self):
            plt.subplot(2,2,1)      
            # plt.plot(x,y)
            plt.plot(self.__binMeanError)
            plt.title("test")
            plt.xlabel("Epsilon")
            plt.ylabel("EpsilonError")
            plt.show()

            #plt.clf()
            plt.subplot(2,2,2)
            plt.plot(self.__binMeanError) #.
            plt.ylabel('number of samples') #.
            plt.show() #.

            #plt.clf()
            plt.subplot(2,2,3)
            plt.plot(self.__binMeanError) #.
            plt.ylabel('min error') #.
            plt.show() #.

            #plt.clf()
            plt.subplot(2,2,4)
            plt.plot(self.__binMeanError) #.
            plt.ylabel('max error') #.
            plt.show() #.
            #print("mean", allDiffs/j)

    class Bin:
        def __init__(self):
            self.value=0
            self.count=0
        
    
    def __init__(self, binCount):
        self.binAmmount=binCount
        self.bin=self.Bin()
        self.bins=[]
        self.statistics = self.Statistics(binCount)

        for i in range(self.binAmmount):
            internalBin=self.Bin()
            self.bins.append(internalBin)

    def CreateBins(self):
        for i in range(self.binAmmount):
            internalBin=self.Bin()
            self.bins.append(internalBin)

    def AddCalculation(self, epsilon, error):
        self.bins[self.__currentBinNumber(epsilon)].value+=error
        self.bins[self.__currentBinNumber(epsilon)].count+=1
        self.statistics.addErrorsToStatistic(epsilon, error, self.__currentBinNumber(epsilon))

    def __currentBinNumber(self, epsilon):
        return int(epsilon/(1/self.binAmmount))


# stats = Stats(binCount=100)
# for a in np.arange(0.0,1.0,0.01):
#     stats.AddCalculation(a, a+0.7)
# stats.statistics.plotStatistics()


# fasdfas=123