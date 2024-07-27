

class Rectangle:
    def __init__(self, top_left, bottom_right, weight):
        self.top_left = top_left

        # self.bottom_right = (
        #     bottom_right[0] if bottom_right[0] < 24 else 23,
        #     bottom_right[1] if bottom_right[1] < 24 else 23
        # )
        self.bottom_right = bottom_right
        self.weight = weight
