import numpy as np
import math

"""
The conversion between rotation matrix and Euler Angles.

"""

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)

    return n < 1e-6


def r2e(R):
    # assert (isRotationMatrix(R))

    # sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[0, 1] * R[0, 1])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[1, 2], R[2, 2])
        y = math.atan2(-R[0, 2], sy)
        z = math.atan2(R[0, 1], R[0, 0])
    else:
        x = math.atan2(-R[2, 1], R[1, 1])
        y = math.atan2(-R[0, 2], sy)
        z = 0

    return np.array([x, y, z])


def e2r(theta):
    """
    Convert Euler Angles to rotation matrix.
    :param theta: Euler Angles, in radians.
    :return: rotation matrix. An 3 x 3 array.
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), math.sin(theta[0])],
                    [0, -math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, -math.sin(theta[1])],
                    [0, 1, 0],
                    [math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), math.sin(theta[2]), 0],
                    [-math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_x, np.dot(R_y, R_z))

    return R


# Calculates Rotation Matrix given euler angles.
def e2r_t(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

if __name__ == "__main__":
    print(e2r([np.pi/2, np.pi/3, np.pi/6]))
    print(e2r_t([np.pi/2, np.pi/3, np.pi/6]))

    rot = [[0.4330127,  0.7500000,  0.5000000],
           [0.2500000,  0.4330127, -0.8660254],
           [-0.8660254,  0.5000000,  0.0000000]]

    print(r2e(np.array(rot).T))
    print(np.pi/2, np.pi/3, np.pi/6)