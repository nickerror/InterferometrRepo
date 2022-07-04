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