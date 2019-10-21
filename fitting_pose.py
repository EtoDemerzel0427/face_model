import numpy as np
from rotations import r2e


def fitting_pose(pt3d, pt2d):
    """
    Solve the linear equation for estimating pose：
    pt4d' * T = pt2d' ===> T = （pt4d * pt4d')^(-1) * pt4d * pt2d'
    :param pt3d: 3 x 87 array. The 3D coordinates of landmarks.
    :param pt2d: 2 x 87 array. The correspondence 2D coordinates of landmarks.
    :return:
    rot: The rotation matrix.
    the projection is an orthographic, so that we can just remove the last colomun
    of the 3D coordinates.
    f: A scalar, should be something like scale.
    t3d: A (3,) np array.
    """

    assert pt3d.shape[0] == 3
    assert pt2d.shape[0] == 2

    pt4d = np.vstack((pt3d, np.ones((1, pt3d.shape[1]))))
    # TODO: here transform is solved by closed form, using optimization method should be better.
    # transform = np.linalg.inv(pt4d.dot(pt4d.T)).dot(pt4d.dot(pt2d.T)).T
    transform = np.linalg.solve(pt4d.dot(pt4d.T), pt4d.dot(pt2d.T))
    transform = transform.T  # 2 x 4

    # print(transform.shape)
    rot = np.vstack((transform[:, :3], np.ones((1, 3))))
    f_xyz = np.linalg.norm(rot, axis=1)
    # print(rot)
    rot[0, :] = rot[0, :] / f_xyz[0]
    rot[1, :] = rot[1, :] / f_xyz[1]
    rot[2, :] = np.cross(rot[0, :], rot[1, :])
    # print(rot)

    f = (f_xyz[0] + f_xyz[1]) / 2
    theta = r2e(rot)   # TODO: this line will be commented.
    t3d = np.array([transform[0, 3], transform[1, 3], 0])

    #return rot, t3d, f
    return theta, t3d, f


if __name__ == "__main__":
    pt3d = np.random.random((3, 87))
    pt2d = np.random.random((2, 87))
    print(fitting_pose(pt3d, pt2d))
