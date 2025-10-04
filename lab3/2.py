import numpy as np
import  math
import cv2

def gauss_kernel_normalize(size: int, sigma: float):
    kernel = np.zeros( (size, size) )
    center = math.ceil(size / 2)

    for i in range(size):
        for j in range(size):
            kernel[i, j] = math.exp( -1 * ( (i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2)) / (2 * math.pi * sigma ** 2)

    return kernel / np.sum(kernel)

kernel3 = gauss_kernel_normalize(3, 1)
kernel5 = gauss_kernel_normalize(5, 1)
kernel7 = gauss_kernel_normalize(7, 1)
print(kernel3)
print()
print(kernel5)
print()
print(kernel7)