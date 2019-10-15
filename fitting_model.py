import numpy as np


def fitting_model(points, cr, single_value, mu, keypoints, w_id_initial, w_exp_initial, img):
    """

    :param points: 87 x 2 array. Landmarks of current photo/frame.
    :param cr: 34530 x 50 x 47 array. The data of 150 people after PCA.
    :param single_value: 50 x 1 array. Guess should be personal bias vector.
    :param mu: 11510 x 3 array. The blendshape of mean face.
    :param keypoints: 1 x 87 array. The indices of 87 landmarks.
    :param w_id_initial: 47 x 1 array. Initial values for identity parameters.
    :param w_exp_initial: 50 x 1 array. Initial values for expression parameters.
    :param img: The current image/frame.
    :return:
    :w_id: the learned parameters of identities.
    :w_exp: the learned parameters of expressions.
    """
    inner_landmarks = list(range(87))
    w_id, w_exp = w_id_initial, w_exp_initial

    left_vis = list(range(8)) + list(range(15, 66)) + [66, 79, 83, 72]
    right_vis = list(range(7, 15)) + list(range(15, 66)) + [66, 79, 83, 72]

    pt3d = mu[keypoints.reshape(-1), :].T  # 3 x 87
    # todo： fitting left pose and right pose




    return w_id, w_exp
