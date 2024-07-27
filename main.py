import HaarLikeFeature
import Stage
import Rectangle
import xml.etree.ElementTree as ET
from OpenCVClassifierParser import OpenCVClassifierParser
from FaceDetector import FaceDetector
from Utils import load_images

def main():
    pos_training_path = 'training_data/faces'
    neg_training_path = 'training_data/nonfaces'
    pos_testing_path = 'training_data/faces/test'
    neg_testing_path = 'training_data/nonfaces/test'

    # stages = [
    #     Stage(
    #         [
    #             HaarLikeFeature(
    #                 [
    #                     Rectangle((6, 4), (12, 9), -1),
    #                     Rectangle((6, 4), (12, 3), 1)
    #                 ],
    #                 -0.0315119996667,
    #                 2.0875380039215088,
    #                 2.2172100543975830
    #             ),
    #     HaarLikeFeature(
    #                 [
    #                     Rectangle((6, 4), (12, 7), -1),
    #                     Rectangle((10, 4), (4, 7), 1)
    #                 ],
    #                 0.0123960003257,
    #                 -1.8633940219879150,
    #                 1.3272049427032471
    #             )
    #         ]
    #     )
    # ]

    # classifiers_file = ET.parse("data/haarcascade_frontalface_default.xml")
    # root = classifiers_file.getroot()  # opencv_storage
    # for child in root:
    #     print(child.tag, child.attrib)
    # stages = root.find(".//stages")
    # features = root.find(".//features")  # rects
    #
    # for stage in stages:
    #     stageThreshold = stage.find('stageThreshold').text
    #     weakClassifiers = stage.find('weakClassifiers')
    #
    #     for weakClassifier in weakClassifiers:
    #         internalNodes = weakClassifier.find('internalNodes').text.split()
    #         leafValues = weakClassifier.find('leafValues').text.split()
    #
    #         rects = features[int(internalNodes[2])].find('rects')
    #
    #         # test = rects[0].text[:-1].split()
    #
    #         for rect in rects:
    #             rect_parameters = rect.text[:-1].split()

    # parser = OpenCVClassifierParser("data/haarcascade_frontalface_default.xml")
    # stages = parser.parse()

    images = load_images(pos_testing_path)

    # for stage in stages:
    #     print(stage.calculate_prediction(images[0]))

    # for image in images:
    #     result = True
    #
    #     stage_ind = 0
    #     for stage in stages:
    #         if not stage.calculate_prediction(image, (0, 0), 1.0):
    #             result = False
    #             print(stage_ind)
    #             break
    #         stage_ind += 1
    #
    #     print(result)

    face_detector = FaceDetector("data/haarcascade_frontalface_default.xml")

    for image_index in range(0, len(images)):
        print(f"Image: {image_index}")
        face_detector.calculate_prediction(images[image_index])


if __name__ == '__main__':
    main()
