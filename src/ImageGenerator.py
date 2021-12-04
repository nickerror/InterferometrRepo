from PIL import Image
import numpy as np
import math

w = 640
h = 640
epsilon=0.4
SecondRingDiameterPixels=300

#RingsDifference=pow(SecondRingDiameterPixels,2)/(1+sigma)
RingsDifference=w*w/6
#print(RingsDifference)

data=np.ones((w,h), dtype=int)
data=data[:,:]*127

for x in range (w):
  for y in range(h):
    
    data[x,y]=data[x,y]+(127*math.cos( 2*math.pi*( epsilon+ ((pow((x-w/2)*2,2)+pow((y-h/2)*2,2)) /RingsDifference))))
    #print("x: ",x,", y:  ",y,", data:  ",data[x,y], "liczba " , 2*math.pi*( ((pow((x-w/2)*2,2)+pow((y-h/2)*2,2)) /RingsDifference)))
#print(data)
img = Image.fromarray(data)
img.show()

#cv2.imshow("image", data)


