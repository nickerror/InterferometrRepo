import cv2 as cv
def selectImage(img, paint=True):
    if paint:
        cv.imshow("selected to analysis", img)
        cv.waitKey(0)
        #cv.destroyAllWindows() 
    return img
