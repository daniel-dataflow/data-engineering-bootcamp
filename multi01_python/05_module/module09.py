import cv2 as cv
import sys

# pip install opencv-python
# python3 -m pip install opencv-python


img  = cv.imread("python.jpeg")

if img is None:
    sys.exit("file not found!")

cv.imshow("image", img)
cv.waitKey()
cv.destroyWindows()