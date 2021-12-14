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

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 127
      var = 10
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy


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

img=np.array(cv2.imread("0_000_generated.png"))
img=Image.fromarray(noisy("gauss",img).astype(np.uint8))
img.show()










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
