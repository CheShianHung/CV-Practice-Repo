import cv2
image = cv2.imread("../images/grant.jpg")
print(image.shape)
cv2.imshow("Image", image)
cv2.waitKey(0)
