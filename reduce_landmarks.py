import numpy as np
# Reduce the number of landmarks from 137 to 87.

# 87-landmark is a subset of 137-landmark.
pt87_in_pt137 = [15, 16, 17, 18, 19, 20, 21, 0, 1, 2, 3, 4, 5, 6, 7, 129, 130, 131, 132, 133, 134, 135, 136, 121, 122, 123,
                  124, 125, 126, 127,
                  128, 96, 94, 92, 90, 88, 102, 100, 98, 105, 107, 109, 111, 113, 115, 117, 119, 64, 65, 66, 67, 68, 69, 70,
                  71, 72, 73, 74, 75,
                  76, 77, 78, 83, 84, 85, 86, 22, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 48, 62, 60, 58, 56, 54,
                  52, 50]

def get_pt87_from_pt137(pt137):
    ret = []
    for i in range(87): ret.append(pt137[pt87_in_pt137[i]])
    return np.array(ret, dtype=np.float32)

if __name__ == "__main__":
    lds_path = '/Users/momo/Desktop/test_frames/test_video_frames'

    import glob
    import os
    pt_names = sorted(glob.glob(os.path.join(lds_path, 'image-[0-9][0-9][0-9].txt')))

    for pt_name in pt_names:
        new_name = pt_name.split('.')[0] + '-lds87.txt'
        pt_137 = np.loadtxt(pt_name, delimiter=',')
        # print(pt_137.shape)
        pt_87 = pt_137[pt87_in_pt137, :]
        np.savetxt(new_name, pt_87)

