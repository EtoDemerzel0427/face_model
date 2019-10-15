import scipy.io
import os


def load_all_mat(mat_path='./MAT') -> object:
    """
    Load all needed MATLAB mat via scipy.io.loadmat.

    :param mat_path: The directory path of all mats.
    :return: 8 different matrices, now in numpy array format.
    """
    blendshapes = scipy.io.loadmat(os.path.join(mat_path, 'blendshapes.mat'))['Blendshapes'][0][0][0]  # 34530 x 150 x 47

    # pt87.keys() includes 'ParaZK_87', 'index87in186', 'index_new87',
    # only the latter two are used.
    pt87 = scipy.io.loadmat(os.path.join(mat_path, 'pt87.mat'))
    index_new87 = pt87['index_new87']  # 87 x 1
    mean_face = pt87['meanFace']  # 11510 x 3

    # input50_orig47.keys() includes 'Cr', 'U2', 'single_value', 'w_exp_initial', 'w_id_initial'
    # 'U2' is not used.
    input50_ori47 = scipy.io.loadmat(os.path.join(mat_path, 'input50_ori47.mat'))
    cr = input50_ori47['Cr'][0][0][0]  # 34530 x 50 x 47
    single_value = input50_ori47['single_value']  # 50 x 1
    w_exp_initial = input50_ori47['w_exp_initial']  # 47 x 1, first dim = 1, others zeros.
    w_id_initial = input50_ori47['w_id_initial']  # 50 x 1

    triangles = scipy.io.loadmat(os.path.join(mat_path, 'triangles'))['triangles']  # 11400 x 4
    triangles = triangles.astype(int)  # cast to int, since every entry is an index.

    return blendshapes, index_new87, mean_face, cr, single_value, w_exp_initial,  w_id_initial, triangles
