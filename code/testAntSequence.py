import cv2
import argparse
import numpy as np
from numpy import typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from SubtractDominantMotion import SubtractDominantMotion

from LucasKanadeAffine import LucasKanadeAffine, construct_M

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--tolerance',
    type=float,
    default=0.2,
    help='binary threshold of intensity difference when computing the mask',
)
parser.add_argument(
    '--seq_file',
    default='../data/antseq.npy',
)

args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold
tolerance = args.tolerance
seq_file = args.seq_file

seq = np.load(seq_file)

'''
HINT:
1. Create an empty array 'masks' to store the motion masks for each frame.
2. Set the initial mask for the first frame to False.
3. Use the SubtractDominantMotion function to compute the motion mask between consecutive frames.
4. Use the motion 'masks; array for visualization.
'''
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('../data/AntSeq.avi', fourcc, 15, (seq.shape[1], seq.shape[0]))


masks = np.zeros(seq.shape, dtype=bool)

for i in range(seq.shape[2]-1):
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]
    num_iters = 100

    color_img = cv2.cvtColor((255 * It1).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    bin_mask = SubtractDominantMotion(It, It1, threshold, num_iters, tolerance, True)
    # color_mask = cv2.cvtColor((255 * bin_mask).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    full_mask = (255 * bin_mask).astype(np.uint8)
    masked_ants = cv2.bitwise_and(color_img, color_img, mask=~full_mask)
    solid_color = np.zeros((It.shape[1], It.shape[0], 3), np.uint8)
    solid_color[:] = (0, 255, 0)
    color_mask = cv2.bitwise_and(solid_color, solid_color, mask=full_mask)
    final_img = cv2.add(color_mask, masked_ants)
    # cv2.imshow("Color_Mask", final_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # video.write(final_img)

    if i in [30, 60, 90, 120]:
        cv2.imwrite(f"../data/ant_seq_invcomp{i}.jpg", final_img)

cv2.destroyAllWindows()
video.release()


