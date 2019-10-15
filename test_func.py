import unittest
import numpy as np
import math
from load_all_mat import load_all_mat
from rotations import e2r, r2e


class TestFunc(unittest.TestCase):
    def setUp(self) -> None:
        self.blendshapes, self.index_new87, self.mean_face, self.cr, \
        self.single_value, self.w_exp_initial, self.w_id_initial, self.triangles = load_all_mat()

    def test_load_all_mat(self):
        # blendshapes, index_new87, mean_face, cr, single_value, w_exp_initial, w_id_initial = load_all_mat()
        self.assertEqual((34530, 150, 47), self.blendshapes.shape)
        self.assertEqual((87, 1), self.index_new87.shape)
        self.assertEqual((11510, 3), self.mean_face.shape)
        self.assertEqual((34530, 50, 47), self.cr.shape)
        self.assertEqual((50, 1), self.single_value.shape)
        self.assertEqual((47, 1), self.w_exp_initial.shape)
        self.assertEqual((50, 1), self.w_id_initial.shape)
        self.assertEqual((11400, 4), self.triangles.shape)

    def test_e2r(self):
        # the test cases are from https://www.andre-gaschler.com/rotationconverter/
        theta_1 = [np.pi/2, np.pi/3, np.pi/6]
        theta_2 = [0, np.pi/5, np.pi/3]
        theta_3 = [math.radians(103), math.radians(52), math.radians(27.6)]

        res_1 = [[0.4330127,  0.7500000,  0.5000000],
                 [0.2500000,  0.4330127, -0.8660254],
                 [-0.8660254,  0.5000000,  0.0000000]]

        res_2 = [[0.4045085, -0.8660254,  0.2938926],
                 [0.7006293,  0.5000000,  0.5090370],
                 [-0.5877852,  0.0000000,  0.8090170]]

        res_3 = [[0.5456014, 0.7846586, 0.2943299],
                 [0.2852335, 0.1563728, -0.9456159],
                 [-0.7880108, 0.5998821, -0.1384937]]

        self.assertAlmostEqual(0, np.linalg.norm(e2r(theta_1).T - res_1), places=6)
        self.assertAlmostEqual(0, np.linalg.norm(e2r(theta_2).T - res_2), places=6)
        self.assertAlmostEqual(0, np.linalg.norm(e2r(theta_3).T - res_3), places=6)

    def test_r2e(self):
        rot_1 = [[0.4330127,  0.7500000,  0.5000000],
                 [0.2500000,  0.4330127, -0.8660254],
                 [-0.8660254,  0.5000000,  0.0000000]]

        rot_2 = [[0.4045085, -0.8660254, 0.2938926],
                 [0.7006293, 0.5000000, 0.5090370],
                 [-0.5877852, 0.0000000, 0.8090170]]

        rot_3 = [[0.5456014, 0.7846586, 0.2943299],
                 [0.2852335, 0.1563728, -0.9456159],
                 [-0.7880108, 0.5998821, -0.1384937]]

        self.assertAlmostEqual(0, np.linalg.norm(r2e(np.array(rot_1).T) - [np.pi/2, np.pi/3, np.pi/6]), places=6)
        self.assertAlmostEqual(0, np.linalg.norm(r2e(np.array(rot_2).T) - [0, np.pi/5, np.pi/3]), places=6)
        self.assertAlmostEqual(0, np.linalg.norm(r2e(np.array(rot_3).T) - [math.radians(103), math.radians(52), math.radians(27.6)]), places=6)

if __name__ == '__main__':
    unittest.main()
