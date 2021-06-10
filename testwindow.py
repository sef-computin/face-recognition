import cv2
import sys

try:
    cv2.namedWindow('displaymywindows', cv2.WINDOW_NORMAL)
    CV_cat = cv2.imread('bully.jpg')
    cv2.imshow('displaymywindows', CV_cat)
    cv2.waitKey(0)
except:
    e = sys.exc_info()[0]
    print("Error: {}".format(e))

cv2.destroyAllWindows()