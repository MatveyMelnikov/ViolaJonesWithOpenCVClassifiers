from OpenCVClassifierParser import OpenCVClassifierParser


class FaceDetector:
    def __init__(self, classifiers_path):
        parser = OpenCVClassifierParser(classifiers_path)
        self.stages = parser.parse()

    def calculate_prediction(self, int_img):
        min_shape = min(int_img.shape[0], int_img.shape[1])

        for size in range(24, min_shape, 2):
            scale_factor = size / 24
            for y in range(0, int_img.shape[1] - size, 2):
                for x in range(0, int_img.shape[0] - size, 2):
                    result = True

                    stage_ind = 0
                    for stage in self.stages:
                        if not stage.calculate_prediction(int_img, (x, y), scale_factor):
                            print(f"Stop on stage {stage_ind}")
                            result = False
                            break
                        stage_ind += 1
                    if result:
                        print(f"Detect: x: {x}, y: {y}, size: {size}")
