import cv2

image = cv2.imread("charizard.png")
print "shape:"
print image.shape
cv2.imshow("charizard.png", image)

raw = image.flatten()
print "\nraw shape"
print raw.shape
print "raw"
print raw

mean = cv2.mean(image)
print "\nmean"
print mean[:3]

(mean, stds) = cv2.meanStdDev(image)
print "\nmean, standard deviation"
print mean, stds

hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print "\nhist shape"
print hist.shape
print "hist flatten shape"
print hist.flatten().shape

cv2.waitKey(0)
