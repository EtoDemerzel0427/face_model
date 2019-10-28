import numpy as np
from utils.lighting import RenderPipeline
import imageio
import os.path


cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


def visualize(pt3d, faces, index, pic_name, path='render_res'):
    """

    :param pt3d: The 3d coordinates of all points. 11510 x 3.
    :param faces: All faces on the mesh. If sparse, 2850 x 4.
    :param index: The index of front face points.
    :param pic_name: The name of current image.
    :params path: The directory to save rendered images.
    :return:
    """
    assert pt3d.shape == (11510, 3)
    assert faces.shape == (2850, 4)

    new_faces = []
    new_index = set()
    for i in faces:
        if i[0] in index and i[1] in index and i[2] in index and i[3] in index:
            new_index.update([i[0], i[1], i[2], i[3]])
            new_faces.append(i)

    new_index = list(new_index)
    mapping = dict(zip(new_index, list(range(len(new_index)))))

    for face in new_faces:
        for i, j in enumerate(face):
            face[i] = mapping[j]

    faces = np.stack(new_faces, axis=0)
    vertices = pt3d[new_index, :]
    # print('faces.shape:', faces.shape)
    # print('faces max:', np.max(faces))
    # print('faces min:', np.min(faces))
    # print('vertices.shape', vertices.shape)

    triangles = np.zeros((faces.shape[0] * 2, faces.shape[1] - 1))
    triangles[:faces.shape[0], :] = faces[:, 0:3]
    triangles[faces.shape[0]:, :] = faces[:, (0, 2, 3)]

    triangles = _to_ctype(triangles).astype(np.int32)  # 3 x (2850 x 2)
    vertices = _to_ctype(vertices).astype(np.float32)

    img = imageio.imread(pic_name).astype(np.float32)/255.

    app = RenderPipeline(**cfg)
    img_render = app(vertices, triangles, img)

    pic_path = os.path.join(path, os.path.basename(pic_name))
    imageio.imwrite(pic_path, img_render)
    print(f'writing rendered picture to: {pic_path}')


if __name__ == "__main__":
    vertices = np.load('/Users/momo/Desktop/face_model/vertices.npy')  # m x 3
    faces = np.load('/Users/momo/Desktop/face_model/faces.npy') - 1  # N x 3
    index = np.load('/Users/momo/Desktop/face_model/index.npy')
    print(vertices)

    visualize(vertices, faces, index, pic_name='../face_test/probes/0CC03C16-36B1-F6D7-BA25'
                                               '-DF8E7E941F3820190420_12_0.jpg')
