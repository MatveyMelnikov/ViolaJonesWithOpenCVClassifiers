import IntegralImage as IntegralImage


class HaarLikeFeature(object):
    # 1. list of rectangles, 2. float, 3. float, 4. float
    def __init__(self, areas, threshold, positive_value, negative_value):
        self.areas = areas
        self.threshold = threshold
        self.positive_value = positive_value
        self.negative_value = negative_value

    def get_score(self, integral_image, offset, scale):
        score = 0

        for area in self.areas:
            top_left = (int(area.top_left[0] * scale + offset[0]),
                        int(area.top_left[1] * scale + offset[1]))
            bottom_right = (int(area.bottom_right[0] * scale + offset[0]),
                            int(area.bottom_right[1] * scale + offset[1]))

            score += area.weight * \
                     IntegralImage.sum_region(integral_image, top_left, bottom_right)

        score /= (integral_image.shape[0] * integral_image.shape[1]) * scale

        return score

    def get_vote(self, integral_image, offset, scale):
        score = self.get_score(integral_image, offset, scale)

        return self.positive_value if score > self.threshold else self.negative_value
