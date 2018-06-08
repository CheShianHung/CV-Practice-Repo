from pyimagesearch.nn import NeuralNetwork
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", default="20000", help="number of epochs to train the neural network")
args = vars(ap.parse_args())

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=int(args["epochs"]))

for (x, target) in zip(X, y):
  pred = nn.predict(x)[0][0] # [0][0] to get the predict value from the array
  step = 1 if pred > 0.5 else 0
  print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target, pred, step))

