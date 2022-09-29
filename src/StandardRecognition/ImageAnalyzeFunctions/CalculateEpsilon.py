from cProfile import label
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import stats
from ImageAnalyzeFunctions.CalculateMiddle import FindMiddle
from ImageAnalyzeFunctions.MaximsFunctions import CalculateMaxims

def CalculateEpsilon(img, widthStripesAmount, heightStripesAmount, diagnose):
    #diagnose=0
    
    #middleWidth=320
    #middleHeight=240

    middleWidth, middleHeight=FindMiddle(img, diagnose=0)
    #vertMean=CircleMean(img, middleWidth, middleHeight)
    #plt.plot(vertMean)
    #plt.show()
    if(diagnose==1):print("Middles: ", middleWidth, middleHeight)

    stripeWidth=img.shape[1]
    stripe=img[(middleHeight-20):(middleHeight+20),:]
    imgResizedPlot=cv2.resize(stripe, (int(stripeWidth), 1),  interpolation = cv2.INTER_AREA)[0,:]

    maxims=CalculateMaxims(stripe=stripe, stripewidth=stripeWidth, middleWidth=middleWidth, windowWidthEdge= 5,windowWidthMiddle= 13, diagnose=diagnose, meaningRadius=19)

    if(diagnose==1):
        plt.plot(stripe.mean(0), label="Uśredniony przebieg")
        for maxim in maxims:
            plt.axvline(maxim,0,1, color='r')#, label="Wyznaczone pozycje prążków")
        plt.axvline(maxims[0],0,1, color='r', label="Wyznaczone pozycje prążków")

    if(diagnose==1):
        print("maxims",maxims)

    #StripMiddle=FindingMaximsMid(maxims=maxims, diagnose=diagnose)
    StripMiddle=middleWidth
    #StripMiddle=320
    #middleHeight=240

    halfMaximsR=[i for i in maxims if i>=StripMiddle ]
    halfMaximsL=[i for i in maxims if i<=StripMiddle ]
    halfMaxims=[]
    if(StripMiddle<(stripeWidth/2)): halfMaxims=halfMaximsR
    else:
        for val in halfMaximsL:
            element=2*StripMiddle-val
            halfMaxims.insert(0, element)

    if(halfMaxims[1]-halfMaxims[0]<halfMaxims[2]-halfMaxims[1]):
        halfMaxims[1]=(halfMaxims[0]+halfMaxims[1])/2
        halfMaxims.pop(0)
    for halfMaxim in halfMaxims:
        if(diagnose==1): plt.axvline(halfMaxim,0.25,0.75, color='g')#, label='Prążki wykorzystane do późniejszych obliczeń')
    if(diagnose==1): plt.axvline(halfMaxims[0],0.25,0.75, color='g', label='Współrzędne wykorzystane do późniejszych obliczeń')
    if(diagnose==1): 
        print("StripMiddle",StripMiddle)
        print("halfMaxims",halfMaxims)
        plt.axvline(StripMiddle, 0.25, 0.75, color='b', label='Wyznaczony środek obrazu')
        plt.xlabel("Numer piksela")
        plt.ylabel("Intensywność")
        plt.legend(loc="lower left")
        plt.show()
    if(diagnose==1):
        #plt.plot(stripe.mean(0), color='r', label='Oryginalne wartości')
        imgResized=copy.deepcopy(stripe)
        meanedStripe=cv2.medianBlur(imgResized,11)
        meanedStripe=meanedStripe.mean(0)
        plt.plot(meanedStripe, color='g', label='Uśredniony przebieg')
        plt.xlabel("Numer piksela")
        plt.ylabel("Intensywność")
        for maxim in maxims:
            plt.axvline(maxim,0,1, color='b')
        plt.axvline(maxims[0],0,1, color='b', label='Położenie prążka')
        plt.legend()
        plt.show()
    halfMaxims=[(i-StripMiddle) for i in halfMaxims]
    if(diagnose==1):print("halfMaxims",halfMaxims)
    if(halfMaxims[0]<halfMaxims[1]-halfMaxims[0]):
        halfMaxims.pop(0)
    halfMaxims=[(i*i) for i in halfMaxims]
    if(diagnose==1):print("halfMaxims",halfMaxims)
    
    halfMaxims=halfMaxims[0:min(len(halfMaxims),4)]
    pointsAmmount=len(halfMaxims)
    x=np.arange(1,pointsAmmount+1,1)
    
    slope, intercept, r, p, std_err = stats.linregress(x, halfMaxims)
    halfMaxims.insert(0,0)
    a=slope
    b=intercept
    
    epsilon=abs(1-(-b/a))
    calculatedPlot=[]
    for i in range (len(halfMaxims)):
        calculatedPlot.append(i*a+b)
    if(diagnose==1):
        print("a:", a, "b:", b)
        plt.plot(halfMaxims, label="Kwadraty odległości od prążków")
        plt.plot(calculatedPlot, c='r', label="Prosta wyznaczona regresją liniową")
        plt.xlabel("Indeksy prążków")
        plt.ylabel("Kwadraty odległości od środka")
        plt.legend()#loc="upper left")
        plt.show()
    
    return epsilon, int(StripMiddle), middleHeight