from random import randint

import cv2
import numpy as np

img = cv2.imread(r'image.jpg')
cv2.imshow('output', img)
cv2.waitKey(0)


img = cv2.imread(r'image.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('output2', img)
cv2.waitKey(0)

img = cv2.imread(r'image.jpg', cv2.IMREAD_COLOR_BGR)
cv2.imshow('output2', img)
cv2.waitKey(0)

img = cv2.imread(r'image.jpg', cv2.IMREAD_ANYCOLOR)
cv2.imshow('output2', img)
cv2.waitKey(0)

#Тут кроче флаги
img = cv2.imread(r'image.jpg', cv2.WINDOW_AUTOSIZE)
cv2.imshow('output2', img)
cv2.waitKey(0)

img = cv2.imread(r'image.jpg', cv2.WINDOW_FULLSCREEN)
cv2.imshow('output2', img)
cv2.waitKey(0)

img = cv2.imread(r'image.jpg', cv2.WINDOW_GUI_NORMAL)
cv2.imshow('output2', img)
cv2.waitKey(0)
