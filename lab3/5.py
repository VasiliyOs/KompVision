import cv2
import math
import numpy as np

img = cv2.imread("img.png")

new_img = cv2.GaussianBlur(img, (15,15), 5)

cv2.imwrite("inner_blur.png", new_img)
