import cv2 as cv
import numpy as np
def closingCircle(img, radius = 5, paint = False):
    kernel = np.ones((radius,radius), np.uint8)
    imgReturn = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    
    if paint:
        cv.imshow("after closing", img)
        cv.waitKey(0)
        #cv.destroyAllWindows() 
    return imgReturn

def openingCircle(img, radius = 5, paint = False):
    kernel = np.ones((radius,radius), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    if paint:
        cv.imshow("after opening", img)
        cv.waitKey(0)
        #cv.destroyAllWindows() 
    return img