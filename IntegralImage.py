import numpy as np


def to_integral_image(img_arr):
    integral_image_arr = np.zeros((img_arr.shape[0], img_arr.shape[1]))
    for x in range(img_arr.shape[1]):
        row_sum = 0

        for y in range(img_arr.shape[0]):
            value = int(img_arr[y, x])

            integral_image_arr[y, x] = \
                (integral_image_arr[y, x - 1] if x > 0 else 0) + row_sum + value

            row_sum += value

    return integral_image_arr


def sum_region(integral_img_arr, top_left, bottom_right):
    # swap tuples
    top_left = (top_left[1], top_left[0])
    bottom_right = (bottom_right[1], bottom_right[0])

    if top_left == bottom_right:
        return integral_img_arr[top_left]

    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    return integral_img_arr[bottom_right] - integral_img_arr[top_right] -\
        integral_img_arr[bottom_left] + integral_img_arr[top_left]
