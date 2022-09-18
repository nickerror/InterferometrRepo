from cProfile import label
from pyexpat import model
import numpy as np
from matplotlib import pyplot as plt
class Stats:

    class Statistics:

        class __ReturnedValues:
            def __init__(self):
                self._outputs = []
                self._labels = []
                self._lossCalculatedError = []

        def __init__(self, chartName, binCount = 100):
            self.__chartName = chartName
            self.__binCount = binCount
            self.__minError = 1.0
            self.__maxError = 0.0
            self.__binMinError = np.ones([binCount,1],dtype=float)
            self.__binMaxError = np.zeros([binCount,1],dtype=float) 
            self.__binMeanError = np.zeros([binCount,1], dtype=float)
            self.__samplesQuantityBin = np.zeros([binCount,1], dtype=float)

            self.__yAxis = np.zeros([binCount,1], dtype=float)
            for x in range(binCount):
                self.__yAxis[x] = x/binCount
            self.__returnedValues = self.__ReturnedValues()

        
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

        def addReturnedStatistics(self, output, label, lossCalculatedError):
            self.__returnedValues._outputs.append(output)
            self.__returnedValues._labels.append(label)
            self.__returnedValues._lossCalculatedError.append(lossCalculatedError)


        def plotStatistics(self):
            fig, imgplot = plt.subplots(2,2)

            imgplot[0][0].plot(self.__yAxis, self.__binMeanError/self.__samplesQuantityBin)
            imgplot[0][0].set_title("Średni błąd bezwzględny dla danego zakresu Eps")
            imgplot[0][0].set_xlabel("Epsilon")
            imgplot[0][0].set_ylabel("Błąd bezwzględny")


            imgplot[0][1].plot(self.__yAxis, self.__samplesQuantityBin)
            imgplot[0][1].set_title("Liczba próbek dla danego zakresu Eps")
            imgplot[0][1].set_xlabel("Epsilon")
            imgplot[0][1].set_ylabel("Liczba próbek")


            imgplot[1][0].plot(self.__yAxis, self.__binMinError)
            imgplot[1][0].set_title("Minimalny błąd dla danego zakresu Eps")
            imgplot[1][0].set_xlabel("Epsilon")
            imgplot[1][0].set_ylabel("min. błąd")


            imgplot[1][1].plot(self.__yAxis, self.__binMaxError)
            imgplot[1][1].set_title("Maksymalny błąd dla danego zakresu Eps")
            imgplot[1][1].set_xlabel("Epsilon")
            imgplot[1][1].set_ylabel("max. błąd")

            fig.suptitle(self.__chartName)
            plt.show() 

        def plotReturnedStatistics(self):
            plt.clf()
            plt.cla()

            plt.plot(self.__returnedValues._outputs, 'b', label = 'outputs')
            plt.plot(self.__returnedValues._labels, 'g', label = 'labels')
            plt.plot(self.__returnedValues._lossCalculatedError, 'r', label = 'error from loss func.')

            plt.legend(loc='lower right')
            plt.show()

    class Bin:
        def __init__(self):
            self.value=0
            self.count=0
        
    
    def __init__(self, binCount, chartName = "model"):
        self.binAmmount=binCount
        self.bin=self.Bin()
        self.bins=[]
        self.statistics = self.Statistics(chartName, binCount)

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

