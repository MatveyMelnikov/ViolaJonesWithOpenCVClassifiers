import xml.etree.ElementTree as ET
from Rectangle import Rectangle
from HaarLikeFeature import HaarLikeFeature
from Stage import Stage


class OpenCVClassifierParser:
    def __init__(self, path):
        self.features = None
        self.stages = None
        self.root = None
        self.classifiers_file = ET.parse(path)

    def parse(self):
        current_stages = []

        self.root = self.classifiers_file.getroot()  # opencv_storage
        self.stages = self.root.find(".//stages")
        self.features = self.root.find(".//features")  # rects

        for stage in self.stages:
            current_stages.append(self.parse_stages(stage))

        return current_stages

    def parse_stages(self, stage):
        stage_threshold = stage.find('stageThreshold').text
        weak_classifiers = stage.find('weakClassifiers')
        weak_classifiers_in_stage = []

        for weak_classifier in weak_classifiers:
            weak_classifiers_in_stage.append(
                self.parse_weak_classifiers(weak_classifier)
            )

        return Stage(
            weak_classifiers_in_stage,
            float(stage_threshold)
        )

    def parse_weak_classifiers(self, weak_classifier):
        rectangles = []

        internal_nodes = weak_classifier.find('internalNodes').text.split()
        leaf_values = weak_classifier.find('leafValues').text.split()

        rects = self.features[int(internal_nodes[2])].find('rects')

        for rect in rects:
            # x1, y1, delta_x, delta_y, weight
            rect_parameters = rect.text[:-1].split()
            top_left = (int(rect_parameters[0]), int(rect_parameters[1]))

            rectangles.append(
                Rectangle(
                    top_left,
                    (top_left[0] + int(rect_parameters[2]),
                     top_left[1] + int(rect_parameters[3])),
                    float(rect_parameters[4])
                )
            )

        return HaarLikeFeature(
            rectangles,
            float(internal_nodes[3]),
            float(leaf_values[1]),
            float(leaf_values[0])
        )
