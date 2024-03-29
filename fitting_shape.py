import numpy as np

def fitting_shape(exp3d, pt2d, rot, t3d, f, single_value):
    """
    Learn people-wise identity parameters.
    :param exp3d: 261 x 50 array. 261 = 3 x 87, 50 refers to 50 people.
    :param pt2d: 87 x 2 array. Landmarks of current photo/frame.
    :param rot: The 3 x 3 rotation matrix.
    :param t3d: (3,) np array.
    :param f: A scalar. Should be something like scale.
    :param single_value:
    :return:
    """

    assert exp3d.shape == (261, 50)
    exp3d = exp3d.T.reshape(-1, pt2d.shape[0], 3).transpose(2, 1, 0) # 3 x 87 x 50
    exp2d = f * np.tensordot(rot, exp3d, axes=(1, 0))[:2, :, :]  # 2 x 87 x 50, imposing pose
    exp2d = exp2d.swapaxes(0,1).reshape(-1, 50)  # 174 x 50
    # print('exp2d sample:', exp2d[:,18])

    beta = 0.06 * f * f * 500
    regular = np.diag((1/single_value).flatten())
    assert regular.shape == (50, 50)

    t2d = np.tile(t3d[:2], pt2d.shape[0]).reshape(-1, 1)  # 174 x 1
    pt2d = pt2d.reshape(-1, 1)  # 174 x 1

    weight = exp2d.T.dot(exp2d) + beta * regular  # 50 x 50
    y = exp2d.T.dot(pt2d - t2d)

    # w_id = np.linalg.inv(weight).dot(y)
    w_id = np.linalg.solve(weight, y)
    assert w_id.shape == (50, 1)

    return w_id

if __name__ == "__main__":
    exp3d = np.arange(261*50).reshape(261, 50)
    exp2d = np.arange(87*2).reshape(87, 2)
    from rotations import e2r
    rot = e2r([np.pi/2, 0, 0])
    t3d = [1,2,1]
    f = 0.5
    single_value = np.ones((50, 1))
    w_id = fitting_shape(exp3d, exp2d, rot, t3d, f, single_value)
    print('-----------------------Input 1----------------------')
    print('exp3d')
    print(exp3d)
    print('exp2d')
    print(exp2d)
    print('rot')
    print(rot)
    print('t3d')
    print(t3d)
    print('f', f)
    print('---------------------output 1-----------------------')
    print(w_id)