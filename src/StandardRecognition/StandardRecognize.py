import cv2
import matplotlib.pyplot as plt
import time
from ImageManager import ImageManager
from ImageAnalyzeFunctions.CalculateEpsilon import CalculateEpsilon
czas=time.monotonic_ns()
widthStripesAmount=48
heightStripesAmount=64

meanDiff=0
meanSignedDiff=0
count=0
imgManager=ImageManager("generatedNoised")#("generatedNoised") "generatedUnnoised" "real"  "generatedNoisedMiddle"  "generatedUnnoisedMiddle"
diag=0
Diagnose=diag
MiniDiagnose=0
showIMG=diag
Losses=[]
for i in range(20000, 20001, 1):
#for i in range(count):
    if(Diagnose==1):  i=20007 #20007  #200047      #20823   21054    21536   21781  21868  22953  
    count+=1
    img,targetEpsilon=imgManager[i]
    img=cv2.medianBlur(img,21)

    calculatedEpsilon, middleWidth, middleHeight=CalculateEpsilon(img, widthStripesAmount, heightStripesAmount, diagnose=Diagnose)
    #middleWidth, middleHeight=FindMiddle(img, diagnose=0)
    #print(middleWidth, middleHeight)
    windowName=str(i)
    centerCoordinates=(middleWidth, middleHeight)
    radius=10
    color=(255,0,0)
    thickness=-1
    if(Diagnose==1 or showIMG==1):
        img=cv2.circle(img, centerCoordinates, radius, color, thickness )
        img=cv2.circle(img, centerCoordinates, 100, color, 2 )
        img=cv2.circle(img, centerCoordinates, 200, color, 2 )
        img=cv2.circle(img, centerCoordinates, 300, color, 2 )
        
        text="Mid:("+str(middleWidth)+", "+ str(middleHeight)+")"
        img=cv2.putText(img=img, text=text, org=(10,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, color=(255,255,255), thickness=2,  lineType=cv2.LINE_AA)
        print("photo: ", i, "calc", calculatedEpsilon%1, "target: ", targetEpsilon, "diff: ", targetEpsilon-calculatedEpsilon%1)
        if(targetEpsilon-calculatedEpsilon%1>0.1):
            print("Watch me!, photo ", i)
        Losses.append(targetEpsilon-calculatedEpsilon%1)
        cv2.imshow(windowName,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()                        
    else:
        if(abs(targetEpsilon-calculatedEpsilon%1)< (1-abs(targetEpsilon-calculatedEpsilon%1))):
            diff=(targetEpsilon-calculatedEpsilon%1)
        else:diff=(1-abs(targetEpsilon-calculatedEpsilon%1))
        diff-=0.015
        Losses.append(diff)

        meanDiff+=abs(diff)
        if(MiniDiagnose==1):
            print(i, "Mid", str(middleWidth), str(middleHeight), "calc", calculatedEpsilon%1, "target: ", targetEpsilon, "diff: ", diff)
        #if(abs(diff)>0.15):
        #    print("Watch me!, photo ", i, diff)
czas2=time.monotonic_ns()       
print((czas2-czas)/1000000)        
    
print("meanDiff", meanDiff/count)
czas=time.monotonic_ns()
for i in range(20000, 20001, 1):
#for i in range(count):
    if(Diagnose==1):  i=20007 #20007  #200047      #20823   21054    21536   21781  21868  22953  
    count+=1
    img,targetEpsilon=imgManager[i]
    img=cv2.medianBlur(img,21)

    calculatedEpsilon, middleWidth, middleHeight=CalculateEpsilon(img, widthStripesAmount, heightStripesAmount, diagnose=Diagnose)
    #middleWidth, middleHeight=FindMiddle(img, diagnose=0)
    #print(middleWidth, middleHeight)
    windowName=str(i)
    centerCoordinates=(middleWidth, middleHeight)
    radius=10
    color=(255,0,0)
    thickness=-1
    if(Diagnose==1 or showIMG==1):
        img=cv2.circle(img, centerCoordinates, radius, color, thickness )
        img=cv2.circle(img, centerCoordinates, 100, color, 2 )
        img=cv2.circle(img, centerCoordinates, 200, color, 2 )
        img=cv2.circle(img, centerCoordinates, 300, color, 2 )
        
        text="Mid:("+str(middleWidth)+", "+ str(middleHeight)+")"
        img=cv2.putText(img=img, text=text, org=(10,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, color=(255,255,255), thickness=2,  lineType=cv2.LINE_AA)
        print("photo: ", i, "calc", calculatedEpsilon%1, "target: ", targetEpsilon, "diff: ", targetEpsilon-calculatedEpsilon%1)
        if(targetEpsilon-calculatedEpsilon%1>0.1):
            print("Watch me!, photo ", i)
        Losses.append(targetEpsilon-calculatedEpsilon%1)
        cv2.imshow(windowName,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()                        
    else:
        if(abs(targetEpsilon-calculatedEpsilon%1)< (1-abs(targetEpsilon-calculatedEpsilon%1))):
            diff=(targetEpsilon-calculatedEpsilon%1)
        else:diff=(1-abs(targetEpsilon-calculatedEpsilon%1))
        diff-=0.015
        Losses.append(diff)

        meanDiff+=abs(diff)
        if(MiniDiagnose==1):
            print(i, "Mid", str(middleWidth), str(middleHeight), "calc", calculatedEpsilon%1, "target: ", targetEpsilon, "diff: ", diff)
        #if(abs(diff)>0.15):
        #    print("Watch me!, photo ", i, diff)
czas2=time.monotonic_ns()       
print((czas2-czas)/1000000)
print("meanDiff", meanDiff/count)
#plt.plot(Losses)
#plt.grid()
#plt.ylim(-0.15,0.15 )
plt.show()