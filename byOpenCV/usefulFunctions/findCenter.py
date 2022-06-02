from cv2 import HOUGH_GRADIENT
import numpy as np
import cv2 as cv
def centerOfGravity(img, paint = True):
    """Returns center of gravity coordinates
	Args:
	img: image (as binnary img)

	Return:
	cX: X coordinate 
    cY: Y coordinate 
	"""
    M = cv.moments(img)

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    if paint:
        cv.circle(img, (cX, cY), 14, (255, 255, 255), -1)
        cv.circle(img, (cX, cY), 10, (50, 255, 100), -1)
        centerText = "Center" + str(cX) + ", " + str(cY)
        cv.putText(img,centerText , (cX - 40, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 1.0,(255, 255, 255), 10)
        cv.putText(img,centerText , (cX - 40, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 1.0,(50, 255, 100), 6)

        # display the image
        cv.imshow("Center of gravity", img)
        cv.waitKey(0)
        cv.destroyAllWindows() 

    return cX, cY

def houghMethod(img, paint = True, _dp=1.0, _minDist=20.0, _param1 = 100.0, _param2 = 80.0, _minRadius = 0.0, _maxRadius=0.0):
    """Returns center of Hough coordinates
	Args:
	img: image (as binnary img)

	Return:
	"""

    rows = img.shape[0]
    cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

    #circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,rows/8,param1=100,param2=80,minRadius=0,maxRadius=0)
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT, _dp,_minDist,param1=_param1,param2=_param2,minRadius=_minRadius,maxRadius=_maxRadius)
   

    try:
        circles = np.uint16(np.around(circles))

        #    if paint:
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        cv.imshow('detected circles',cimg)
        cv.waitKey(0)
    except:
        cv.putText(cimg,"Not found circles" , ( 40, 25),cv.FONT_HERSHEY_SIMPLEX, 1.0,(255, 255, 255), 12)
        cv.putText(cimg,"Not found circles" , ( 40, 25),cv.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255), 6)
        cv.imshow('not found circles',cimg)
        cv.waitKey(0)
