from FaceDetector import FaceDetector
from Utils import load_images


def main():
    pos_training_path = 'training_data/faces'
    neg_training_path = 'training_data/nonfaces'
    pos_testing_path = 'training_data/faces/test'
    neg_testing_path = 'training_data/nonfaces/test'

    images = load_images(pos_testing_path)
    face_detector = FaceDetector("data/haarcascade_frontalface_default.xml")

    for image_index in range(0, len(images)):
        print(f"Image: {image_index}")
        faces = face_detector.detect(images[image_index], 2.0, 1.25, 0.1, 3)
        for face in faces:
            print(f"Detect - x: {face.x}, y: {face.y}, size: {face.size}")


if __name__ == '__main__':
    main()
