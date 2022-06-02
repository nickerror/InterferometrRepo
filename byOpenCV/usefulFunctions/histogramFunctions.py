from cv2 import calcHist
from matplotlib import pyplot as plt

def calcHistogram(img, paint = False):
    """Return and Paint histogram
    Args:
    img: image to threshold
    paint: if true, painting will be display

    Return:
    hist - histogram
    """
    hist = calcHist([img],[0],None,[256],[0,256])
    if paint:
        plt.plot(hist)
        plt.xlim([0,256])
        plt.show()

    return hist