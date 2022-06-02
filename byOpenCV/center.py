from contextlib import closing
import cv2 as cv

from usefulFunctions import crop, thresholdFunctions, histogramFunctions, findCenter, morphology,anotherFunctions


paintAll = True

img = cv.imread("C:\\Users\\Krzysztof\\Documents\\Studia_informatyka\\praca_magisterska\\InterferometrRepo\\data\\raw\\1channel\\photo\\real\\abc.png", cv.IMREAD_GRAYSCALE)
img = cv.medianBlur(img, 5)

img = crop.centerCrop(img, (488,488))

hist = histogramFunctions.calcHistogram(img, paint = paintAll)

threshBinary, threshBinaryINV, threshTrunc, threshToZero, threshToZeroINV = thresholdFunctions.basicTresholds(img, paint = paintAll)
adaptiveMeanThreshold, adaptiveGausianThreshold = thresholdFunctions.adaptiveTresholds(img, paint = paintAll)

############################################################################################################
###########################################___FIND_CENTER___#################################################
#############################################################################################################
paintAll = True
#will use adaptiveGausianThreshold 

selectedImage = anotherFunctions.selectImage(adaptiveMeanThreshold, paint=paintAll) #original
#selectedImage = anotherFunctions.selectImage(cv.medianBlur(selectedImage,5), paint=paintAll)

closedImage = morphology.openingCircle(selectedImage, radius=5, paint = paintAll)
openedImage = morphology.closingCircle(closedImage, radius=5, paint = paintAll)

#selectedImage = anotherFunctions.selectImage(openedImage, paint=paintAll)

cX, cY = findCenter.centerOfGravity(selectedImage, paint = False)


rows = selectedImage.shape[0]

findCenter.houghMethod(selectedImage,paint=False,
    _dp = 1,
    _minDist = 10,
    _param1 = 100,
    _param2 = 80,
    _minRadius = 50,
    _maxRadius = 400
    )


#Brudnopis

edges = cv.Canny(openedImage,100,255)
cv.imshow("edges canny", edges)
cv.waitKey(0)

im2, contours, hierarchy = cv.findContours(openedImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.imshow("conturs", im2)
cv.waitKey(0)

# findCenter.houghMethod(selectedImage,paint=True,\
#     _dp = 1.0,
#     _minDist = 0.1,
#     _param1 = 250.0, #krawędzie
#     _param2 = 90.0,  #jaki zajebisty jest okrąg
#     _minRadius = 0,
#     _maxRadius = 400
#     )


# findCenter.houghMethod(selectedImage,paint=True,\
#     _dp = 1,
#     _minDist = 0.1,
#     _param1 = 100,
#     _param2 = 80,
#     _minRadius = 0,
#     _maxRadius = 400
#     )
