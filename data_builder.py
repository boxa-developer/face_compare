import dlib
import numpy as np
import timeit
import os
import argparse
from scipy.spatial import distance

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--predictor', required=False,
                default='models/shape_predictor_68_face_landmarks.dat',
                help='Predictor File Path')
ap.add_argument('-m', '--model', required=False,
                default='models/dlib_face_recognition_resnet_model_v1.dat',
                help='Recognition Model File Path')
args = vars(ap.parse_args())


def to_numpy_array(vector):
    array = []
    for i in range(0, len(vector)):
        array.append(vector[i])
    return array


def compute_similarity(x1, x2):
    x = distance.euclidean(x1, x2)
    return 100 * round(1 / (1 + x), 2)


class FaceCompare:
    def __init__(self):
        super(FaceCompare, self).__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(args['predictor'])
        self.face_rec = dlib.face_recognition_model_v1(args['model'])

    def compute_descriptor(self, filename):
        img = dlib.load_rgb_image(filename)
        detections = self.detector(img, 1)
        # shape = self.predictor(img, detections[0])
        faces = dlib.full_object_detections()
        for detection in detections:
            faces.append(self.predictor(img, detection))
        face_ds = []
        for i in range(len(faces)):
            tmpArr = to_numpy_array(self.face_rec.compute_face_descriptor(img, faces[i]))
            face_ds.append(tmpArr)
        print("Computed!")
        return face_ds

#
# s = timeit.default_timer()
# obj = FaceCompare()
# e = timeit.default_timer()
# scanObj = os.scandir('faces')
# print(f'Faces Directory Loaded in {e - s}')
# s = timeit.default_timer()
# i = 0
# for item in scanObj:
#     start = timeit.default_timer()
#     file_name = os.path.join('faceData', item.name.split('.')[:-1][0] + '.npy')
#     farr = obj.compute_descriptor(os.path.join('faces', item.name))
#     # np.save(file_name, farr)
#     i += 1
#     stop = timeit.default_timer()
#     # print(f'Generated {file_name} in {stop - start}')
# e = timeit.default_timer()
# print(f'Overall Time: {e - s} Count: {i}')

# np.save('f1.npy',obj.computeDescriptor('./faces/f1.jpg'))
# arr = np.load('f1.npy')
