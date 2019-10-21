import numpy as np
from scipy.optimize import minimize

def fitting_expression(id3d, pt2d, rot, t3d, f, w_exp):
    """

    :param id3d: 261 x 47 array. 261 = 3 x 87, 47 refers to 47 expressions.
    :param pt2d: 87 x 2 array. Landmarks of current photo/frame.
    :param rot: 3 x 3 Rotation matrix from pose estimation.
    :param t3d: （3,) np array. Estimated via fitting_pose.
    :param f: A scalar. Estimated via fitting_pose.
    :param w_exp: 47 x 1 np array. The initial values for optimization.
    :return: the optimized w_exp.
    """

    w_exp.flatten()
    test_a = id3d[3:6, :]
    id3d = id3d.T.reshape(-1, pt2d.shape[0], 3).transpose(2, 1, 0)   # 3 x 87 x 47
    test_b = id3d[:, 1, :].squeeze()
    assert np.linalg.norm(test_a - test_b) < 1e-6   # test if reshaping is correct

    weight2d = f * np.tensordot(rot, id3d, axes=(1, 0))[:2, :, :]  # 2 x 87 x 47
    weight2d = weight2d.swapaxes(0,1).reshape(-1, 47)  # 174 x 47

    mean_weight = weight2d[:, 0]  # (174,)
    weight = weight2d[:, 1:]  # 174 x 46

    t2d = np.tile(t3d[:2], pt2d.shape[0]).flatten()  # (174,)
    pt2d = pt2d.flatten()  # 174 x 1

    # x should be (46,) , the expression parameters w/o the mean expression(always 1).
    # quadratic problem with L1 norm.
    beta = 0.01
    fun = lambda x : np.sum(np.square(weight.dot(x) + mean_weight + t2d - pt2d)) + beta * np.sum(np.abs(x))

    bounds = [(0,1)] * (weight2d.shape[1] - 1)  # each element should be in (0,1)

    # initial = w_exp[1:]
    initial = np.zeros(46)
    res = minimize(fun, initial, bounds=bounds)

    return np.concatenate(([1], res.x), axis=0)









