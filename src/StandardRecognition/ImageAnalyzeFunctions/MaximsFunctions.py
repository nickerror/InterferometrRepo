import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy


def MeaningMaxims(maxims, meanedStripe):
    maximsInt=np.array(np.round(maxims), dtype=int)
    maximsVal=[meanedStripe[val] for val in maximsInt]
    maximsMean=np.mean(maximsVal)
    maximsFiltered=[]
    
    for i in range(len(maxims)):
        if(maximsVal[int(i)]>0.81*maximsMean or ( maxims[int(i)]>540) ):
            maximsFiltered.append(maxims[i])

    diffs=np.diff(maximsFiltered)
    indToDelete=[]
    #indToMean=[i for i in range(1,len(diffs)-1, 1) if(not(diffs[i-1]>diffs[i] and diffs[i]<diffs[i+1]))]
    for i in range(1,len(diffs)-1, 1):
        if((diffs[i-1]>diffs[i]*1.3 and diffs[i]*1.3<diffs[i+1]) and diffs[i]-diffs[i-1]<20):
            indToDelete.append(i)
    
    if(len(indToDelete)>0):
        indToDelete.sort(reverse=True)
        for i in indToDelete:
            indLeft=i
            indRight=i+1
            indMidd=(indLeft+indRight)/2
            if(meanedStripe[int(indMidd)]>(0.95*(meanedStripe[int(indLeft)]+meanedStripe[int(indRight)])/2)):
                maximsFiltered[i]=(maximsFiltered[i]+maximsFiltered[i+1])/2
                maximsFiltered.pop(i+1)
    return maximsFiltered

def CalculateMaxims(stripe, stripewidth, middleWidth=-1, windowWidthEdge=0, windowWidthMiddle=0, meaningRadius=11, diagnose=0, threshold=10):
    if(diagnose==1): 
        print(stripe.shape)
        print("CalculateStripMiddle")
    imgResized=copy.deepcopy(stripe)
    imgResizedSize=imgResized.shape[1]

    if(middleWidth==-1):
        middleWidth=stripewidth/2  #Niewiadomo, dlatego centralnie środek
        
    meanedStripe=cv2.medianBlur(imgResized,meaningRadius)
    #meanedStripe=scipy.interpolate.interp1d(imgResized,num=21)
    meanedStripe=meanedStripe.mean(0)
    
    if(diagnose==1):
        imgResized=imgResized.mean(0)
        plt.plot(imgResized, color='r', label='Oryginalne wartości')
        plt.plot(meanedStripe, color='g', label='Uśredniony przebieg')
        plt.xlabel("Numer piksela")
        plt.ylabel("Intensywność")
        plt.legend()
        plt.show()

    maxims=[]
    lastMaxim=0
    if(diagnose==1):
        print("Calculating new strip")
    for i in range (windowWidthMiddle, (imgResizedSize-windowWidthMiddle-1), 1):
        windowWidth=round(max(windowWidthMiddle-abs((middleWidth-i)/middleWidth*(windowWidthMiddle-windowWidthEdge)),windowWidthEdge))
        #print(windowWidth)
        if(meanedStripe[i]==max(meanedStripe[i-windowWidth:i+windowWidth])):
            if(len(maxims)>0 and int((maxims[-1]+windowWidth))<i):
                maxims.append(i)
                lastMaxim=i
            elif(len(maxims)==0):
                maxims.append(i)
            elif(int((maxims[-1]+windowWidth))>i):
                maxims[-1]=(lastMaxim+i)/2
   
    maximsFiltered=MeaningMaxims(maxims, meanedStripe)
    #maximsMean=np.mean([(meanedStripe[int(i)]) for i in maxims])
    #maximsFilteredMean=np.mean([meanedStripe[int(i)] for i in maximsFiltered])
    
    if(diagnose==1):
        print("maximsUnfiltered", maxims)
        print("maximsFiltered",maximsFiltered)
    
    #if(maximsMean*0.988>maximsFilteredMean):
    #    maximsFiltered=maxims

    
    return maximsFiltered


def FindingMaximsMid(maxims, diagnose):
    diffs=np.diff(maxims)
    if(diagnose==1): print("DiffsUnfiltered", diffs)
    ind=np.argpartition(diffs,-3)[-3:]

    if(diagnose==1):
        print("maxims", maxims)
        print("diffs", diffs)
        print("ind", ind)
        print(diffs[ind[2]])
    StripMiddle=0
    shift=1
    diffsInd=[diffs[ind[0]],diffs[ind[1]], diffs[ind[2]]]
    if(abs(ind[0]-ind[1])==2 and abs(ind[0]-ind[2])==1 and abs(ind[1]-ind[2])==1 and (diffs[ind[1]]/diffs[ind[0]])<1.4):
        StripMiddle=(maxims[min(ind)]+maxims[max(ind)+1])/2
        if(diagnose==1): print("Selected_  1")
    elif(abs(ind[2]-ind[1])==2):
        if((diffs[ind[1]]/diffs[ind[0]])<1.1 and abs(ind[1]-ind[0])==3):
            ind_edge=ind[:2]
            StripMiddle=(maxims[min(ind_edge)]+maxims[max(ind_edge)+1])/2 
            if(diagnose==1): print("Selected_  2.2")
        elif (max(diffsInd)<(min(diffsInd))*1.2):
            StripMiddle=(maxims[min(ind)]+maxims[max(ind)+1])/2 
            if(diagnose==1): print("Selected_  2.1")
        else:
            ind_edge=ind[1:]
            StripMiddle=(maxims[min(ind_edge)]+maxims[max(ind_edge)+1])/2   
            if(diagnose==1): print("Selected_  2")
    elif(abs(ind[2]-ind[0])==2 and diffs[ind[2]]/diffs[ind[0]]<1.5):
        StripMiddle=(maxims[min(ind)]+maxims[max(ind)+1])/2   
        if(diagnose==1): print("Selected_  3")
    elif((diffs[ind[1]]/diffs[ind[0]])>1.5 and (diffs[ind[2]]/diffs[ind[1]])>1.4):
        StripMiddle=(maxims[min(ind[2], ind[1])]+maxims[max(ind[1], ind[2])+1])/2   
        if(diagnose==1): print("Selected_  3.5")
    elif(abs(ind[2]-ind[1])==1 and diffs[ind[1]]*1.4>diffs[ind[2]]):
        ind_edge=ind[1:]
        StripMiddle=(maxims[min(ind_edge)]+maxims[max(ind_edge)+1])/2  
        if(diagnose==1): print("Selected_  4")
    elif(abs(ind[2]-ind[1])==1 and abs(ind[1]-ind[0])==1 and abs(ind[2]-ind[0])==2):
        if(maxims[ind[2]]/maxims[ind[0]]>1.4):
            StripMiddle=(maxims[ind[2]]+maxims[ind[2]+1])/2
            if(diagnose==1): print("Selected_  5.1")
        else:
            StripMiddle=(maxims[min(ind)]+maxims[max(ind)])/2
            if(diagnose==1): print("Selected_  5")
    elif(abs(ind[2]-ind[1])==1 and diffs[ind[1]]<=1.2*diffs[ind[2]]):
        #StripMiddle=(maxims[min(ind)]+maxims[max(ind)+1])/2
        StripMiddle=(maxims[ind[2]]+maxims[ind[2]+1])/2
        if(diagnose==1): print("Selected_  6.1")
    elif(abs(ind[2]-ind[1])==1 and diffs[ind[1]]<=1.2*diffs[ind[2]]):
        #StripMiddle=(maxims[min(ind)]+maxims[max(ind)+1])/2
        StripMiddle=(maxims[ind[2]]+maxims[ind[2]+1])/2
        if(diagnose==1): print("Selected_  6")
    elif(abs(ind[2]-ind[1])>2):
        ind_edge=ind[1:]
        StripMiddle=(maxims[min(ind_edge)]+maxims[max(ind_edge)+1])/2  
        if(diagnose==1): print("Selected_  7")


    elif(abs(ind[0]-ind[1])==1):
        print("NewSituatiooooooooooooooooooooooooooooooooooon")
        ind0Val=abs( maxims[ind[0]]-maxims[ind[0]+1] )
        ind1Val=abs( maxims[ind[1]]-maxims[ind[1]+1] )
        if (ind0Val>ind1Val and ind0Val>1.5*ind1Val):#  or  (ind1Val>0 and ind1Val>2*ind0Val)): 
            StripMiddle=(maxims[ind[0]-shift]+maxims[ind[0]+1+shift])/2
        elif (ind1Val>ind0Val and ind1Val>1.5*ind0Val):
            StripMiddle=(maxims[ind[1]-shift]+maxims[ind[1]+1+shift])/2
        else:
            ind=np.sort(ind)
            StripMiddle=(maxims[ind[0]-shift]+maxims[ind[1]+1+shift])/2
    elif(abs(ind[0]-ind[1])==2):
        print("NewSituatiooooooooooooooooooooooooooooooooooon")
        ind=np.sort(ind)
        StripMiddle=(maxims[ind[0]-shift]+maxims[ind[1]+1+shift])/2
    else:
        print("NewSituatiooooooooooooooooooooooooooooooooooon")
        StripMiddle=(maxims[ind[0]-shift]+maxims[ind[0]+1]+shift)/2

    #print(abs(ind[0]-ind[1]), abs(ind[0]-ind[2]), abs(ind[1]-ind[2]) )
    


    return StripMiddle