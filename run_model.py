import argparse
from load_all_mat import load_all_mat
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

blendshapes, index_new87, mean_face, cr, single_value, w_exp_initial,  w_id_initial, triangles = load_all_mat(args['path'])

# down sample the mesh
faces_load = triangles[:, 2].reshape(-1, 4)  # 2850 x 4
indices = index_new87.T  # 1 x 87

pic_names, pt_names = None, None
height, width, nchannels = None, None, None
if args["camera"]:
    # TODO: read frame from video stream
    from imutils.video import VideoStream
    import time
    vs = VideoStream(src=0).start()
    time.sleep(2.0)  # allow sensors to warm up
    frame = vs.read()
    height, width, nchannels = frame.shape
else:
    root_path = '../face_test/probes'
    pic_names = sorted(glob.glob(os.path.join(root_path, '*.jpg')))
    pt_names = sorted(glob.glob(os.path.join(root_path, '*lds87.txt')))

    first_img = cv2.imread(os.path.join(root_path, pic_names[0]))
    height, width, nchannels = first_img.shape  # 256 x 256 x 3

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# Define the fps to be equal to 5 if use camera, 30 if not. Also frame size is passed.
if args['camera']:
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (width, height))
else:
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

if args['camera']:
    pass
else:
    for i in range(1, len(pic_names)):
        img = cv2.imread(pic_names[i])
        points = np.loadtxt(pt_names[i], delimiter=',')  # 87 x 2

        points[:, 1] = height + 1 - points[:, 1]

        # TODO: FittingModel

        # TODO: draw mesh










