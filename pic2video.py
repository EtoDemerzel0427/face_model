import os.path as osp
import sys
from glob import glob
import imageio

def pic2video(pic_path, video_path=None):
    if video_path is None:
        video_path = pic_path

    # fps = sorted(glob(osp.join(pic_path, '*.jpg')))
    fps = sorted(glob(osp.join(pic_path, '*.jpeg')))

    imgs = []
    for fp in fps:
        img = imageio.imread(fp)
        imgs.append(img)

    imageio.mimwrite(osp.basename(video_path) + '.mp4', imgs, fps=24, macro_block_size=None)



if __name__ == "__main__":
    pic2video("render_res")
