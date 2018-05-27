from scipy.spatial import distance as dist
import numpy as np
np.random.seed(42)
x = np.random.rand(4)
y = np.random.rand(4)

print "x: "
print x
print "y: "
print y

print "euclidean: "
print dist.euclidean(x, y)
print "cityblock: "
print dist.cityblock(x, y)
print "chebyshev: "
print dist.chebyshev(x, y)

x = np.random.random_integers(0, high = 1, size = (4, ))
y = np.random.random_integers(0, high = 1, size = (4, ))
print "\nnew x: "
print x
print "new y: "
print y

print "hamming: "
print dist.hamming(x, y)

