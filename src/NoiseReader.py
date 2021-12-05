from PIL import Image
import cv2
import numpy as np
import math
import os
from numpy import loadtxt
import glob
import random
def OldReadNoise():
    Combined=np.zeros((480,640))
    FotoCount=300
    for i in range(FotoCount):
        Read=np.array(cv2.imread("data/interim/"+str(i).zfill(5)+".png"))
        Read=Read[4:484,4:644,0]
        ReadMean=np.mean(Read)
        Combined=(Combined+Read)#-ReadMean
        if (i%100==0): print(i)
    Combined=Combined/FotoCount
    Combined=Combined.astype(np.uint8)

    Noise=Image.fromarray(Combined)
    Noise.save("Szum.png")

def ReadNoise():
    #FotoCount=100
    #randEpsilons=np.random.rand(FotoCount).round(decimals=3)
    randEpsilons=np.arange(0,0.999,0.001)
    FotoCount=len(randEpsilons)
    path="data/external/"
    Combined=np.zeros((480,640))
    i=0
    for randEps in randEpsilons:
         
        if (i%50==0):
            print(i)
        epsilon=str(format(randEps,'.3f')).replace('.','_')
        epsilonFileNames=glob.glob(path+epsilon + "*.png", recursive = True)
        if(len(epsilonFileNames)>0):
            filenumber=random.randint(0,len(epsilonFileNames)-1)
            Read=np.array(cv2.imread(epsilonFileNames[filenumber]))
            print(epsilonFileNames[filenumber])
            Read=Read[4:484,4:644,0]
            ReadMean=np.mean(Read)
            Combined=(Combined+Read)#-ReadMean
        else: 
            FotoCount=FotoCount-1
        i=i+1
    Combined=Combined/FotoCount
    Combined=Combined.astype(np.uint8)
    Noise=Image.fromarray(Combined)
    Noise.save("SZUUUUM.png")


def Unnoising():
    Noise=np.array(cv2.imread("Szum.png"))
    Noise=Noise[:,:,0]
    NoiseMean=np.mean(Noise)
    print(NoiseMean)
    Differ=-(Noise-NoiseMean)

    Eps0_52=np.array(cv2.imread("data/external/0_000__01463.png"))
    Eps0_52=Eps0_52[4:484,4:644,0]
    Eps0_52=Eps0_52+Differ
    Eps0_52=np.clip(Eps0_52,0,255)

    #Eps0_52 = cv2.fastNlMeansDenoising(Eps0_52,None,10,10,14,42)

    #kernel = np.ones((2,2), np.uint8)


    #Eps0_52 = cv2.dilate(Eps0_52,kernel, iterations=1)
    #Eps0_52 = cv2.erode(Eps0_52,kernel, iterations=1)
    Eps0_52=Eps0_52.astype(np.uint8)
    #Eps0_52 = cv2.bilateralFilter(Eps0_52,9,75,75)
    #Eps0_52 = cv2.erode(Eps0_52,kernel, iterations=1)
    #Eps0_52 = cv2.dilate(Eps0_52,kernel, iterations=1)
    #Eps0_52 = cv2.bilateralFilter(Eps0_52,9,75,75)


    Unnoised=Image.fromarray(Eps0_52)
    Unnoised.save("0_000__01463_Unnoised.png")

def Noising():
    Generated=np.array(cv2.imread("data/generated/0_000.png"))
    Noise=np.array(cv2.imread("SZUUUUM.png"))
    Noise=Noise[:,:,0]
    NoiseMean=np.mean(Noise)
    print(NoiseMean)
    Differ=-(Noise-NoiseMean)
    Generated=Generated[:,:,0]
    Generated=Generated-Differ
    Generated=np.clip(Generated,0,255)
    Generated=Generated.astype(np.uint8)
    Noised=Image.fromarray(Generated)
    Noised.save("NoisedImage.png")

def ImagesRename():
    lines=loadtxt('data/external/c211021.txt')

    for i in range(lines.shape[0]):
        
        oldname='data/external/'+str(i).zfill(5)+'.png'
        newname='data/external/'+ str(format(lines[i,1].astype(float),'.3f')).replace('.','_')+'__'+str(i).zfill(5)+'.png'
        if (i%100==0) : print('o:', oldname,'   n: ', newname)
        
        os.rename(oldname,newname)

Unnoising()










#Eps0_52=np.array(cv2.imread("data/interim/00009.png"))
#Eps0_00=np.array(cv2.imread("data/interim/19323.png"))
##Eps0_52mean=np.mean(Eps0_52)
#Eps0_00mean=np.mean(Eps0_00)

#Combined=Eps0_52+Eps0_00
#Noise=(Combined-Eps0_00mean-Eps0_52mean)/2
#Noise=Noise.astype(np.uint8)


#Combined=np.array(cv2.imread("data/interim/00000.png"))
#for i in range(5000):
#print(Combined.shape)
#Combined=Combined.astype(np.uint8)
#Combined=Combined[4:484,4:644,0]
#print(Combined.shape)
##img = Image.fromarray(Combined)
#cv2.imshow("Szum", Combined)
#cv2.waitKey(0) # waits until a key is pressed
#cv2.destroyAllWindows() # destroys the window showing image
#img.save('____0.png')

#New=np.array(cv2.imread("data\processed\\0_000.png"))
#print(New.shape)
#Eps0=np.array(cv2.imread("data/interim/00000.png"))
#Eps0_00=np.array(cv2.imread("data/interim/19323.png"))
#Eps0_52mean=np.mean(Eps0_52)
#Eps0_00mean=np.mean(Eps0_00)
#Combined=Eps0_52+Eps0_00
#Noise=(Combined-Eps0_00mean-Eps0_52mean)/2
#Noise=Noise.astype(np.uint8)
