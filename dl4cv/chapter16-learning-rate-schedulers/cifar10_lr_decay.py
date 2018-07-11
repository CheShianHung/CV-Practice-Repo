import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

def step_decay(epoch):
  initAlpha = 0.01
  factor = 0.25
  dropEvery = 5

  alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

  return float(alpha)

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
ap.add_argument("-m", "--model", required=True, help="path to the output model")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# step = int(len(testX) / 10)
# 
# trainX = trainX[::step]
# trainY = trainY[::step]
# testX = testX[::step]
# testY = testY[::step]

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["ariplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

callbacks = [LearningRateScheduler(step_decay)]

opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, callbacks=callbacks, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])

model.save(args["model"])

# ((trainX, trainY), (testX, testY)) = cifar10.load_data()
# 
# testX2 = testX.astype("float") / 255.0
# 
# testY = lb.transform(testY)
# 
# step = int(len(testX) / 10)
# 
# testX = testX[::step]
# testX2 = testX2[::step]
# testY = testY[::step]
# 
# preds = model.predict(testX2).argmax(axis=1)
# for (i, image) in enumerate(testX):
#   image = np.array(image)
# #  cv2.putText(image, "Label: {}".format(labelNames[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#   cv2.imshow("Image", image)
#   cv2.waitKey(0)

