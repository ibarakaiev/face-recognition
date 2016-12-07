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

    # Lines 29 through 48 divide all the facial points into separated sections
    # in order to simplify array for graphical use (not included in demo examples in repo)

    # 0-10
    jaw_points = numpy.array([np_shape[0], np_shape[2], np_shape[3], np_shape[4], np_shape[6],
                              np_shape[8], np_shape[10], np_shape[12], np_shape[13], np_shape[14], np_shape[16]])
    # 11-13
    right_brow_points = numpy.array([np_shape[17], np_shape[19], np_shape[21]])
    # 14-16
    left_brow_points = numpy.array([np_shape[22], np_shape[24], np_shape[26]])
    # 17-21
    nose_points = numpy.array([np_shape[27], np_shape[29], np_shape[31], np_shape[33], np_shape[35]])
    # 22-28
    right_eye_points = numpy.array(np_shape[36:42])
    # 29-35
    left_eye_points = numpy.array(np_shape[42:48])
    # 36-42
    outer_mouth_points = numpy.array([np_shape[48], np_shape[50], np_shape[51], np_shape[52], np_shape[54],
                                      np_shape[55], np_shape[57], np_shape[59]])
    # 43-51
    inner_mouth_points = numpy.array(np_shape[60:68])
    shape = numpy.concatenate([jaw_points, right_brow_points, left_brow_points,
                               nose_points, right_eye_points, left_eye_points,
                               outer_mouth_points, inner_mouth_points])
    """
    shape_graphed = []
    i = 0
    while i < shape.size/2:
        shape_graphed.append(shape[i])
        j = i+1
        closest = []
        while j < shape.size/2:
            dist = math.sqrt(math.pow((shape[i, 0]-shape[j, 0]), 2) + math.pow((shape[i, 1 ]-shape[j, 1]), 2))
            if dist < 70:
                closest.append(shape[j])
            j += 1
        j = 0
        while j < len(closest)-1:
            shape_graphed.append(closest[j])
            shape_graphed.append(closest[j+1])
            shape_graphed.append(shape[i])
            j += 1
        i += 1

    shape_graphed_np = numpy.array(shape_graphed)
    """

    plt.scatter(shape[:, 0], shape[:, 1], c='w', s=8)
    # plt.plot(shape_graphed_np[:, 0], shape_graphed_np[:, 1], c='w')

plt.show()
dlib.hit_enter_to_continue()
