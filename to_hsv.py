import cv2
import numpy as np

img = cv2.imread('image.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("Baza", img)
cv2.imshow("HSV", hsv)

cv2.waitKey(0)