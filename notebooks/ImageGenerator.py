from PIL import Image
import cv2
import numpy as np
import math


def main():
  w = 640 #width
  h = 480 #height
  middleX=w/2 #Middle of centre, X axis
  middleY=h/2 #Middle of centre, Y axis
  ringsDifference=w*w/6.07 #Difference in squared distance from center between succesive stripes
  #57 600 #68 266
  #epsilon=0 #set epsilon
  epsilons=np.arange(1.0)/1000.0

  pixelMax=210  #max brightness of pixel, 8bit-> 0-255
  pixelMin=40 #min brightness of pixe, 8bit-> 0-255

  for epsilon in epsilons:
    img=GenerateImage(w,h,middleX,middleY,ringsDifference,epsilon,pixelMax,pixelMin)
    print(img.shape)
    img = Image.fromarray(img)
    
    path="data/processed/"+str(("%.3f" %epsilon).replace('.','_').replace(' ','a'))+str(".png")
    print(path)
    img.save(path)
    
  
  #img.show()
  #image=np.array(img)

  #cv2.imshow('image',image)
  #cv2.imshow("Nazwa", data)
  #cv2.waitKey(0) # waits until a key is pressed
  #cv2.destroyAllWindows() # destroys the window showing image

  #cv2.imshow("image", data)


def GenerateImage(w,h,middleX,middleY,ringsDifference,epsilon,pixelMax,pixelMin):
  pixelMean=(pixelMax+pixelMin)/2
  pixelDiff=pixelMax-pixelMean

  data=np.ones((h,w))
  data=data[:,:]*pixelMean
  for x in range (w):
    for y in range(h):
      data[y,x]=data[y,x]+(pixelDiff*math.cos( 2*math.pi*( epsilon+ ((pow((x-middleX)*2,2)+pow((y-middleY)*2,2)) /ringsDifference))))
    #print("x: ",x,", y:  ",y,", data:  ",data[x,y], "liczba " , 2*math.pi*( ((pow((x-w/2)*2,2)+pow((y-h/2)*2,2)) /RingsDifference)))
  data=data.astype(np.uint8)
  return data

main()