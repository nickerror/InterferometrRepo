import cv2
import matplotlib.pyplot as plt
from ImageAnalyzeFunctions.MaximsFunctions import CalculateMaxims
from ImageAnalyzeFunctions.MaximsFunctions import  FindingMaximsMid


def WidthMiddleFind(img, middleHeight, precalculatedMiddle=-1, windowWidthEdge=2, windowWidthMiddle=3, diagnose=0):
    if(diagnose==1): print(img.shape)
    stripeWidth=img.shape[1]
    stripe=img[(middleHeight-40):(middleHeight+40),:]

    maxims=CalculateMaxims(stripe=stripe, stripewidth=stripeWidth, middleWidth=precalculatedMiddle, windowWidthEdge= windowWidthEdge,
                            windowWidthMiddle= windowWidthMiddle, diagnose=diagnose, meaningRadius=19)
    widthMiddle = FindingMaximsMid(maxims=maxims, diagnose=diagnose)

    if(diagnose==1):
        plt.plot(stripe.mean(0))
        for maxim in maxims:
            plt.axvline(maxim,0,1, color='r')
        plt.axvline(widthMiddle, 0.25, 0.75, color='b', label='Wyznaczony środek obrazu')
        plt.xlabel("Numer piksela")
        plt.ylabel("Intensywność")
        plt.show()
    if(diagnose==1): print("CalculatedMiddle", widthMiddle)
    return int(widthMiddle)

def FindMiddle(img, diagnose=0):
    if(diagnose==1): print("Finding width middle")
    middleWidth=WidthMiddleFind(img=img, middleHeight=240, diagnose=diagnose)
    if(diagnose==1): print("Finding height middle")
    middleHeight=WidthMiddleFind(img=cv2.transpose(img), middleHeight=middleWidth, windowWidthEdge=2, windowWidthMiddle=13, diagnose=diagnose)
    if(diagnose==1): print("Finding width middle")
    middleWidth=WidthMiddleFind(img=img, middleHeight=middleHeight, precalculatedMiddle=middleWidth, windowWidthEdge=3, windowWidthMiddle=5, diagnose=diagnose)
    return middleWidth, middleHeight


