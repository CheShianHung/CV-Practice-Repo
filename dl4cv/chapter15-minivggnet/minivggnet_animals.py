from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from keras.models import load_model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

classLabels = ["cat", "dog", "panda"]

#print("[INFO] loading images...")
#imagePaths = list(paths.list_images(args["dataset"]))
#
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
#
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
#(data, labels) = sdl.load(imagePaths, verbose=500)
#data = data.astype("float") / 255.0
#
#(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
#
#trainY = LabelBinarizer().fit_transform(trainY)
#testY = LabelBinarizer().fit_transform(testY)
#
#print("[INFO] compiling model...")
#opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
#model = MiniVGGNet.build(width=32, height=32, depth=3, classes=3)
#model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#
#H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, verbose=1)
#
#print("[INFO] serializing network...")
#model.save(args["model"])
#
#print("[INFO] evaluating network...")
#predictions = model.predict(testX, batch_size=32)
#print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))
#
#plt.style.use("ggplot")
#plt.figure()
#plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
#plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
#plt.title("Training Loss and Accuracy")
#plt.xlabel("Epoch #")
#plt.ylabel("Loss/Accuracy")
#plt.legend()
#plt.show()


print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

preds = model.predict(data, batch_size=32).argmax(axis=1)

for (i, imagePath) in enumerate(imagePaths):
  image = cv2.imread(imagePath)
  cv2.putText(image, "Label: {}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
  cv2.imshow("Image", image)
  cv2.waitKey(0)
