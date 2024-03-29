import argparse
from load_all_mat import load_all_mat
from fitting_model import fitting_model
from visualize import visualize
import cv2
import os
import glob
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--camera", type=bool, default=False,
                help="whether to use the webcam or frame files")
ap.add_argument("-p", "--path", type=str, default="./MAT",
                help="path to the mat files")
args = vars(ap.parse_args())

print("[Working] Loading data...")
blendshapes, index_new87, mean_face, cr, single_value, w_exp_initial, w_id_initial, triangles = load_all_mat(
    args['path'], from_npy=True)
print("[Finished] Data loaded.")


# down sample the mesh
faces_load = triangles[:, 2].reshape(-1, 4)  # 2850 x 4
indices = index_new87.T  # 1 x 87
indices = indices - 1  # IMPORTANT: convert MATLAB 1-based index to python 0-based index

pic_names, pt_names = None, None
height = None
if args["camera"]:
    # TODO: read frame from video stream
    from imutils.video import VideoStream
    import time

    vs = VideoStream(src=0).start()
    time.sleep(2.0)  # allow sensors to warm up
    frame = vs.read()
    height, width, nchannels = frame.shape
else:
    # root_path = '../face_test/probes'
    root_path = '/Users/momo/Desktop/test_frames/test_video_frames'
    # pic_names = sorted(glob.glob(os.path.join(root_path, '*.jpg')))
    pic_names = sorted(glob.glob(os.path.join(root_path, '*.jpeg')))
    pt_names = sorted(glob.glob(os.path.join(root_path, '*lds87.txt')))
    print(f'[Counted] Total pic number is {len(pic_names)}')

    first_img = cv2.imread(pic_names[0])
    height, width, nchannels = first_img.shape  # 256 x 256 x 3

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# Define the fps to be equal to 5 if use camera, 30 if not. Also frame size is passed.
# if args['camera']:
#     out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (width, height))
# else:
#     out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

wd = 'render_res'
if not os.path.exists(wd):
    os.mkdir(wd)

if args['camera']:
    pass
else:
    for i in range(len(pic_names)):
        # img = cv2.imread(pic_names[i])
        # print(pic_names[i])

        print(f'[Processing] pic number {i+1}...')
        # points = np.loadtxt(pt_names[i], delimiter=',')  # 87 x 2
        points = np.loadtxt(pt_names[i])  # 87 x 2


        points[:, 1] = height + 1 - points[:, 1]  # reverse for historical reason.

        # 1. learn identity and expression coefficients
        f, rot, t3d, w_id, w_exp = fitting_model(points, cr, single_value, indices, w_id_initial,
                                                   w_exp_initial)

        # 2. predict 3d mesh for new img
        pt3d_predict = np.tensordot(cr, w_exp, axes=(2, 0)).squeeze()  # 34530 x 50, apply expression
        pt3d_predict = np.tensordot(pt3d_predict, w_id, axes=(1, 0)).reshape(-1, 3).T   # 3 x 11510, apply shape

        # pt3d_predict = f * np.tensordot(rot, pt3d_predict, axes=(1,0)).T  # 11510 x 3, apply pose
        pt3d_predict = f * np.dot(rot, pt3d_predict).T + np.tile(np.reshape(t3d, (1, -1)), (pt3d_predict.shape[1], 1))


        # 3. draw mesh, borrow code from: https://github.com/cleardusk/3DDFA
        index = np.where(pt3d_predict[:, 2] > -10)[0]  # front face vertices
        visualize(pt3d_predict, faces_load - 1, index, pic_names[i])


