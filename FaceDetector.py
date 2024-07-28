from OpenCVClassifierParser import OpenCVClassifierParser
import IntegralImage as IntegralImage


class FaceDetector:
    def __init__(self, classifiers_path):
        parser = OpenCVClassifierParser(classifiers_path)
        self.stages = parser.parse()

    def detect(
            self,
            image,
            base_scale,
            scale_increment,
            position_increment,
            min_neighbors
    ):
        integral_image = IntegralImage.to_integral_image(image)
        integral_image_squares = IntegralImage.to_integral_image_of_squares(image)
        found_faces = []  # Rectangles - (x, y, size)

        # min_shape = min(image.shape[0], image.shape[1])
        #
        # for size in range(24, min_shape + 1, 2):
        #     scale_factor = size / 24
        #     for y in range(0, image.shape[1] - size + 1, 2):
        #         for x in range(0, image.shape[0] - size + 2, 2):
        #             result = True
        #
        #             stage_ind = 0
        #             for stage in self.stages:
        #                 if not stage.calculate_prediction(integral_image, integral_image_squares, (x, y), scale_factor):
        #                     print(f"Stop on stage {stage_ind}")
        #                     result = False
        #                     break
        #                 stage_ind += 1
        #             if result:
        #                 print(f"Detect: x: {x}, y: {y}, size: {size}")

        max_scale = min(image.shape[0] / 24.0, image.shape[1] / 24.0)

        for scale in \
                map(
                    lambda n: n / 10.0,
                    range(int(base_scale * 10), int(max_scale * 10), int(scale_increment * 10))
                ):
            step = int(scale * 24 * position_increment)
            size = int(scale * 24)

            for x in range(0, image.shape[0] - size, step):  # We do not consider the right edge
                for y in range(0, image.shape[1] - size, step):  # We do not consider the bottom edge
                    result = True

                    # stage_ind = 0
                    for stage in self.stages:
                        if not stage.calculate_prediction(integral_image, integral_image_squares, (x, y), scale):
                            # print(f"Stop on stage {stage_ind}")
                            result = False
                            break
                        # stage_ind += 1
                    if result:
                        # print(f"Detect: x: {x}, y: {y}, size: {size}")
                        found_faces.append((x, y, size))

        return self.merge(found_faces, min_neighbors)

    def merge(self, found_faces, min_neighbors):
        result = []
        similar_areas = []

        for area_ind in range(0, len(found_faces)):
            similar_to_current = [found_faces[area_ind]]

            for comparison_area_ind in range(area_ind + 1, len(found_faces)):
                if self.is_similar(found_faces[area_ind], found_faces[comparison_area_ind]):
                    similar_to_current.append(found_faces[comparison_area_ind])

            similar_areas.append(similar_to_current)

        for neighbours in similar_areas:
            if len(neighbours) >= min_neighbors:
                result.append(self.combine_neighbours(neighbours))

        return result


    def is_similar(self, first_area, second_area):
        delta = first_area[2] * 0.2

        delta_x = abs(first_area[0] - second_area[0])
        delta_y = abs(first_area[1] - second_area[1])
        delta_size = abs(first_area[2] - second_area[2])

        if delta_x <= delta and delta_y <= delta and delta_size <= delta:
            return True
        else:
            return False

    def combine_neighbours(self, neighbours):
        result_x = 0
        result_y = 0
        result_size = 0

        for neighbour in neighbours:
            result_x += neighbour[0]
            result_y += neighbour[1]
            result_size += neighbour[2]

        result_x /= len(neighbours)
        result_y /= len(neighbours)
        result_size /= len(neighbours)

        return result_x, result_y, result_size

