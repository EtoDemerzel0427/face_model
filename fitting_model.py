import numpy as np
from fitting_pose import fitting_pose
from fitting_expression import fitting_expression
from rotations import e2r


def fitting_model(pt2d, cr, single_value, mu, keypoints, w_id_initial, w_exp_initial, img):
    """

    :param pt2d: 87 x 2 array. Landmarks of current photo/frame.
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
    # inner_landmarks = list(range(87))
    w_id, w_exp = w_id_initial, w_exp_initial

    # left_vis = list(range(8)) + list(range(15, 66)) + [66, 79, 83, 72]
    # pt3d = mu[keypoints.flatten(), :]
    # print(pt3d[left_vis, :].shape)
    # theta, t3d, f, trans = fitting_pose(pt3d[left_vis, :].T, pt2d[left_vis, :].T)
    # rot = e2r(theta)
    # right_vis = list(range(7, 15)) + list(range(15, 66)) + [66, 79, 83, 72]

    pt3d = mu[keypoints.reshape(-1), :].T  # 3 x 87
    # todo： fitting left pose and right pose

    keys = np.vstack((3 * keypoints - 2, 3 * keypoints - 1, 3 * keypoints)).T
    assert keys.shape == (87, 3)
    keys = keys.flatten()
    cr = cr[keys, :, :]  # 261 x 50 x 47

    theta, t3d, f = None, None, None
    for i in range(5):
        tmp = np.tensordot(cr, w_id, axes=(1,0))  # 261 x 47 x 1
        pt3d = np.tensordot(tmp, w_exp, axes=(1,0)).reshape(3, -1)  # 261 x 1 x 1 ===> 3 x 87

        # 1. pose estimation
        theta, t3d, f, transform = fitting_pose(pt3d, pt2d.T)
        rot = e2r(theta)

        # 2. expression estimation
        id3d = np.tensordot(cr, w_id, axes=(1,0)).squeeze()  # 261 x 47
        w_exp = fitting_expression(id3d, pt2d, rot, t3d, f, w_exp).reshape(-1, 1)

        # 3. shape estimation
        # TODO： add shape estimation



        print('[cur_res] w_id: ', w_id)
        print('[cur_res] w_exp: ', w_exp)



    return f, theta, t3d, w_id, w_exp
