import IntegralImage as IntegralImage
import math


class HaarLikeFeature:
    # 1. list of rectangles, 2. float, 3. float, 4. float
    def __init__(self, areas, threshold, left_value, right_value):
        self.areas = areas
        self.threshold = threshold
        self.left_value = left_value
        self.right_value = right_value

    def get_score(self, integral_image, offset, scale):
        score = 0

        for area in self.areas:
            top_left = (int(area.top_left[0] * scale + offset[0]),
                        int(area.top_left[1] * scale + offset[1]))
            bottom_right = (int(area.bottom_right[0] * scale + offset[0]),
                            int(area.bottom_right[1] * scale + offset[1]))

            score += area.weight * \
                     IntegralImage.sum_region(integral_image, top_left, bottom_right)

        score /= pow(scale * 24, 2)

        return score

    def get_vote(self, integral_image, integral_image_of_squares, offset, scale):
        score = self.get_score(integral_image, integral_image_of_squares, offset, scale)
        amendment = self.calculate_an_amendment(
            integral_image, integral_image_of_squares, offset, scale
        )

        return self.right_value if score < self.threshold * amendment else self.left_value

    def calculate_an_amendment(self, integral_image, integral_image_of_squares, offset, scale):
        size = 24 * scale
        inv_area = 1 / pow(size, 2)

        top_left = (int(offset[0]),
                    int(offset[1]))
        bottom_right = (int(offset[0] + size),
                        int(offset[1] + size))

        sum_of_pixels = IntegralImage.sum_region(
            integral_image, top_left, bottom_right
        )
        sum_of_squares_of_pixels = IntegralImage.sum_region(
            integral_image_of_squares, top_left, bottom_right
        )

        moy = sum_of_pixels * inv_area
        vnorm = sum_of_squares_of_pixels * inv_area - pow(moy, 2)

        return 1 if vnorm < 1 else math.sqrt(vnorm)
