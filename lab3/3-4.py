import cv2
import math
import numpy as np
from numpy import ndarray


def gauss_kernel_normalize(size: int, sigma: float):
    kernel = np.zeros( (size, size) )
    center = math.ceil(size / 2)

    for i in range(size):
        for j in range(size):
            kernel[i, j] = math.exp( -1 * ( (i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2)) / (2 * math.pi * sigma ** 2)

    return kernel / np.sum(kernel)


def add_pixels(matrix, kernel_size: int):
    padding = kernel_size // 2

    left_columns = [ pixels[: padding]  for pixels in matrix ]
    right_columns = [pixels[len(pixels) - padding :] for pixels in matrix]

    matrix_with_side_columns = []
    for i, row in enumerate(matrix):
        left_pixels = [ pixel for pixel in left_columns[i] ]
        right_pixels = [ pixel for pixel in right_columns[i] ]

        left_pixels.extend(row)
        left_pixels.extend(right_pixels)

        matrix_with_side_columns.append(left_pixels)

    top_rows = matrix_with_side_columns[: padding]
    bottom_rows = matrix_with_side_columns[-padding :]

    new_matrix = top_rows + matrix_with_side_columns + bottom_rows

    return new_matrix



def get_closest_pixels(pixel_coords: tuple, pixels, convulation_matrix_size: int) -> list[list[int]]:
    i, j = pixel_coords

    start_row_index = i - convulation_matrix_size
    end_row_index = i + convulation_matrix_size
    start_column_index = j - convulation_matrix_size
    end_column_index = j + convulation_matrix_size

    result = []
    for row_index in range(start_row_index, end_row_index + 1):
        row_pixels = []
        for column_index in range(start_column_index, end_column_index + 1):
            row_pixels.append(pixels[row_index][column_index])
        result.append(row_pixels)

    return result


def matrix_svertki(matrix: ndarray,
                   kernel: ndarray
                   ):
    convulation_matrix_size = kernel.shape[0] // 2

    extended_matrix = add_pixels(matrix.copy(), kernel.shape[0])
    cv2.imwrite("extended_image.jpg", np.array(extended_matrix, dtype=np.uint8))

    pixels = [
        [
            extended_matrix[i][j]
            for j in range(convulation_matrix_size, len(extended_matrix[0]) - convulation_matrix_size)
        ]
        for i in range(convulation_matrix_size, len(extended_matrix) - convulation_matrix_size)
    ]

    new_matrix = []
    for i, pixel_row in enumerate(pixels):
        new_matrix_row = []
        for j, pixel in enumerate(pixel_row):
            pixel_coords = (i + convulation_matrix_size, j + convulation_matrix_size)
            pixels_around = get_closest_pixels(pixel_coords, extended_matrix, convulation_matrix_size)

            new_pixel_value = np.sum(pixels_around * kernel)
            new_matrix_row.append(new_pixel_value)

        new_matrix.append(new_matrix_row)

    return new_matrix


def gauss_blur(image: str, output_image: str, kernel_size: int, sigma):
    img = cv2.imread(image)

    kernel = gauss_kernel_normalize(kernel_size, sigma)

    b_channel, g_channel, r_channel = cv2.split(img)

    new_b_channel = matrix_svertki(b_channel, kernel)
    new_g_channel = matrix_svertki(g_channel, kernel)
    new_r_channel = matrix_svertki(r_channel, kernel)

    new_b_channel_array = np.array(new_b_channel, dtype=np.uint8)
    new_g_channel_array = np.array(new_g_channel, dtype=np.uint8)
    new_r_channel_array = np.array(new_r_channel, dtype=np.uint8)

    cv2.imwrite(f'{output_image}', cv2.merge([new_b_channel_array, new_g_channel_array, new_r_channel_array]))


gauss_blur("img.png", "5.png", 5, 2)
gauss_blur("img.png", "7.png", 7, 1)