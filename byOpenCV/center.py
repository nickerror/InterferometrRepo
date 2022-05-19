import cv2 as cv

from usefulFunctions import crop, thresholdFunctions, histogramFunctions, findCenter, anotherFunctions


paintAll = False

img = cv.imread("C:\\Users\\Krzysztof\\Documents\\Studia_informatyka\\praca_magisterska\\InterferometrRepo\\data\\raw\\1channel\\photo\\real\\abc.png", cv.IMREAD_GRAYSCALE)
img = cv.medianBlur(img, 5)

img = crop.centerCrop(img, (488,488))

hist = histogramFunctions.calcHistogram(img, paint = paintAll)

threshBinary, threshBinaryINV, threshTrunc, threshToZero, threshToZeroINV = thresholdFunctions.basicTresholds(img, paint = paintAll)
adaptiveMeanThreshold, adaptiveGausianThreshold = thresholdFunctions.adaptiveTresholds(img, paint = paintAll)

############################################################################################################
###########################################___FIND_CENTER___#################################################
#############################################################################################################

#will use adaptiveGausianThreshold 
#selectedImage = anotherFunctions.selectImage(adaptiveGausianThreshold)
selectedImage = anotherFunctions.selectImage(cv.medianBlur(adaptiveGausianThreshold,5))

cX, cY = findCenter.centerOfGravity(selectedImage)

findCenter.houghMethod(selectedImage)