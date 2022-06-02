import cv2 as cv
from matplotlib import pyplot as plt
def basicTresholds(img, valMin = 127, vaalMax = 255, paint = False):
    """Return and Paint different thresholds
    Args:
    img: image to threshold
    valMin: Minimum value to threshold
    valMax: Maximum value to threshold

    Return:
    threshBinary
    threshBinaryINV
    threshTrunc
    threshToZero
    threshToZeroINV
    """
    ret,threshBinary = cv.threshold(img,127,255,cv.THRESH_BINARY) #most useful
    ret,threshBinaryINV = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
    ret,threshTrunc = cv.threshold(img,127,255,cv.THRESH_TRUNC)
    ret,threshToZero = cv.threshold(img,127,255,cv.THRESH_TOZERO)
    ret,threshToZeroINV = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)

    if paint:
        titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
        images = [img, threshBinary, threshBinaryINV, threshTrunc, threshToZero, threshToZeroINV]
        for i in range(6):
            plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()

    return threshBinary, threshBinaryINV, threshTrunc, threshToZero, threshToZeroINV

def adaptiveTresholds(img, paint = False):
    """Return and Paint adaptive thresholds
    Args:
    img: image to adaptive threshold


    Return:
    adaptiveMeanThreshold
    adaptiveGausianThreshold
    """
    ret,binaryThreshold = cv.threshold(img,127,255,cv.THRESH_BINARY)
    adaptiveMeanThreshold = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                cv.THRESH_BINARY,61,2)
    adaptiveGausianThreshold = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,61,2)
    
    if paint:
        titles = ['Original Image', 'Global Thresholding (v = 127)',
                    'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        images = [img, binaryThreshold, adaptiveMeanThreshold, adaptiveGausianThreshold]
        for i in range(4):
            plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()

    return adaptiveMeanThreshold, adaptiveGausianThreshold