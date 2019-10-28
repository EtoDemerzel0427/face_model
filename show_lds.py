import numpy as np
import os.path as osp
import cv2

root_path = '/Users/momo/Desktop/test_frames/test_video_frames'
img = cv2.imread(osp.join(root_path, 'image-150.jpeg'))
pts = np.loadtxt(osp.join(root_path, 'image-150.txt'), delimiter=',')

for point in pts:
    a = int(point[0]), int(point[1])
    cv2.circle(img, a,  1, (0, 255, 0), 1)

cv2.imshow('img', img)
cv2.waitKey()
