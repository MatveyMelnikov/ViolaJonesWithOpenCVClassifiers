import math

import IntegralImage as ii


class HaarLikeFeature(object):
    # 1. list of rectangles, 2. float, 3. float, 4. float
    def __init__(self, areas, threshold, positive_value, negative_value):
        self.areas = areas
        self.threshold = threshold
        self.positive_value = positive_value
        self.negative_value = negative_value

    def get_score(self, int_img, offset, scale):
        score = 0

        for area in self.areas:
            top_left = (int(area.top_left[0] * scale + offset[0]),
                        int(area.top_left[1] * scale + offset[1]))
            bottom_right = (int(area.bottom_right[0] * scale + offset[0]),
                            int(area.bottom_right[1] * scale + offset[1]))

            integral_image = ii.to_integral_image(int_img)
            score += area.weight * \
                     ii.sum_region(integral_image, top_left, bottom_right)

        score /= (int_img.shape[0] * int_img.shape[1]) * scale

        return score

    def get_vote(self, int_img, offset, scale):
        score = self.get_score(int_img, offset, scale)
        #amendment = self.calculate_an_amendment(int_img, offset, scale)

        #return self.positive_value if score > self.threshold * amendment else self.negative_value
        return self.positive_value if score > self.threshold else self.negative_value

    def calculate_an_amendment(self, int_img, offset, scale):
        image_of_squares = ii.to_integral_image_of_squares(int_img)
        integral_image = ii.to_integral_image(int_img)

        top_left = (int(24 * scale + offset[0]),
                    int(24 * scale + offset[1]))
        bottom_right = (int(24 * scale + offset[0]),
                        int(24 * scale + offset[1]))

        sum_of_squares_of_pixels = ii.sum_region(image_of_squares, top_left, bottom_right)
        sum_of_pixels = ii.sum_region(integral_image, top_left, bottom_right)

        moy = sum_of_pixels /\
                (image_of_squares.shape[0] * image_of_squares.shape[1] * scale)
        vnorm = sum_of_squares_of_pixels /\
                (image_of_squares.shape[0] * image_of_squares.shape[1] * scale)
        vnorm -= pow(moy, 2)

        return 1 if vnorm < 1 else math.sqrt(vnorm)
