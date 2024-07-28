from OpenCVClassifierParser import OpenCVClassifierParser
import IntegralImage as IntegralImage
from Area import Area
from Utils import merge


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
        found_faces = []  # Areas

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

                    for stage in self.stages:
                        if not stage.calculate_prediction(
                                integral_image, (x, y), scale
                        ):
                            result = False
                            break

                    if result:
                        found_faces.append(Area(x, y, size))

        return merge(found_faces, min_neighbors)
