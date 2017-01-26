import dlib
from skimage import io
import numpy
import matplotlib.pyplot as plt
import math

# You should download this file manually
predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# feel free to use any photo you want
img = io.imread('pics/mark_zuckerberg.jpg')
plt.imshow(img)

# array of faces
dets = detector(img, 1)
for k, d in enumerate(dets):
    shape = predictor(img, d)
    np_shape = []
    for i in shape.parts():
        np_shape.append([i.x, i.y])
    np_shape = numpy.array(np_shape)

    plt.scatter(np_shape[:, 0], np_shape[:, 1], c='w', s=8)
    # plt.plot(shape_graphed_np[:, 0], shape_graphed_np[:, 1], c='w')

plt.show()
dlib.hit_enter_to_continue()
