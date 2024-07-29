import numpy as np
from PIL import Image
from Area import Area
import os


def load_images(path):
    images = []
    for _file in os.listdir(path):
        if _file.endswith('.jpg'):
            img_arr = np.asarray(Image.open((os.path.join(path, _file))), dtype=np.uint64)

            # Convert RGB to greyscale
            images.append(
                np.transpose(
                    np.floor(np.dot(img_arr[..., :3], [0.3, 0.59, 0.11]))
                )
            )
    return images


def is_similar(first_area, second_area):
    delta = first_area.size * 0.2

    delta_x = abs(first_area.x - second_area.x)
    delta_y = abs(first_area.y - second_area.y)
    delta_size = abs(first_area.size - second_area.size)

    if delta_x <= delta and delta_y <= delta and delta_size <= delta:
        return True
    else:
        return False


def combine_neighbours(neighbours):
    result = Area()

    for neighbour in neighbours:
        result.set(
            result.x + neighbour.x,
            result.y + neighbour.y,
            result.size + neighbour.size
        )

    result.set(
        int(result.x / len(neighbours)),
        int(result.y / len(neighbours)),
        int(result.size / len(neighbours))
    )

    return result


def merge(areas, min_neighbors):
    result = []
    similar_areas = []

    for area_ind in range(0, len(areas)):
        similar_to_current = [areas[area_ind]]

        for comparison_area_ind in range(area_ind + 1, len(areas)):
            if is_similar(areas[area_ind], areas[comparison_area_ind]):
                similar_to_current.append(areas[comparison_area_ind])

        similar_areas.append(similar_to_current)

    for neighbours in similar_areas:
        if len(neighbours) >= min_neighbors:
            result.append(combine_neighbours(neighbours))

    return result

