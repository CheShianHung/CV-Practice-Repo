import imutils
from skimage import exposure
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required = True, help = "Path to the query image")
args = vars(ap.parse_args())

image = cv2.imread(args["query"])
ratio = image.shape[0]/300.0
orig = image.copy()
image = imutils.resize(image, height = 300)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    temp = image.copy()
    #cv2.drawContours(temp, [approx], -1, (0, 255, 0), 3)
    #cv2.imshow("Contour", temp)
    if len(approx) == 4:
        screenCnt = approx
        break
    #cv2.waitKey(0)

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("Game Boy Screen", image)
cv2.waitKey(0)

