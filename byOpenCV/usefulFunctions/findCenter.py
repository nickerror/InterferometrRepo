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

    return cX, cY

def houghMethod(img, paint = True):
    """Returns center of Hough coordinates
	Args:
	img: image (as binnary img)

	Return:
	"""
    cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

    #circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,2,20,param1=50,param2=30,minRadius=0,maxRadius=0)

    circles = np.uint16(np.around(circles))
    
    if paint:
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        cv.imshow('detected circles',cimg)
        cv.waitKey(0)
        cv.destroyAllWindows()