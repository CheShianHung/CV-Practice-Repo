import RGBHistogram
import numpy as np
import argparse
import cPickle
import glob
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to the directory that contains the images we just indexed")
ap.add_argument("-q", "--query", required = True, help = "Path to the query image")
args = vars(ap.parse_args())

index = {}
for imagePath in glob.glob(args["dataset"] + "/*.png"):
    k = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist,hist)
    index[k] = hist.flatten()

queryImage = cv2.imread(args["query"])
queryK = args["query"][args["query"].rfind("/") + 1:]
queryHist = cv2.calcHist([queryImage], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
queryHist = cv2.normalize(queryHist, queryHist)
queryFeature = queryHist.flatten()

results = {}
for (k, features) in index.items():
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + 1e-10) for (a, b) in zip(features, queryFeature)])
    results[k] = d
results = sorted([(v, k) for (k, v) in results.items()])

cv2.imshow("Query", queryImage)
print "query: %s" % (queryK)
montageA = np.zeros((166 * 5, 400, 3), dtype = "uint8")
montageB = np.zeros((166 * 5, 400, 3), dtype = "uint8")
for j in xrange(0, 10):
    (score, imageName) = results[j]
    path = args["dataset"] + "/%s" % (imageName)
    result = cv2.imread(path)
    print "\t%d. %s : %.3f" % (j + 1, imageName, score)
    if j < 5:
        montageA[j * 166:(j + 1) * 166, :] = result
    else:
        montageB[(j - 5) * 166:((j - 5) + 1) * 166, :] = result

cv2.imshow("Results 1-5", montageA)
cv2.imshow("Results 6-10", montageB)
cv2.waitKey(0)
