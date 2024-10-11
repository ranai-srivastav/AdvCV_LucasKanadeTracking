from tempfile import template

import cv2
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanade import LucasKanade, draw_rectangle, add_p_to_rect

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=0.001,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--template_threshold',
    type=float,
    default=5,
    help='threshold for determining whether to update template',
)
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/carseq.npy")
# no_correction = np.load("carseq.npy")

rect = [59, 116, 145, 151]
rects = np.zeros([seq.shape[2], 4])
rects[0] = rect.copy()

prev_p = np.zeros([2, 1])
prev_rect = np.zeros([1, 4])

I0 = seq[:, :, 0]

# for i in tqdm(range(0, np.shape(seq)[2] - 1)):
for i in range(0, np.shape(seq)[2] - 1):
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]

    # rect is updated with values from the prev iteration
    # rect diff from t=0 to now
    p_diff = np.array([rects[i][0] - rects[0][0], rects[i][1] - rects[0][1]]).reshape(2, 1)

    p_n = LucasKanade(It, It1, rects[i], threshold, int(num_iters), p0=prev_p)
    p_n_starred = p_n + p_diff
    p_star = LucasKanade(I0, It1, rects[0], threshold, int(num_iters), p0=p_n_starred.copy())

    if (np.linalg.norm(p_star - p_n_starred)) < template_threshold:
        prev_p = (p_star-p_diff).copy()
        rects[i+1] = np.array(add_p_to_rect(rects[0], p_star)).reshape([1, 4])
    else:
        prev_p = p_n.copy()
        rects[i+1] = np.array(add_p_to_rect(rects[i], p_n)).reshape([1, 4])

    img = draw_rectangle(It1, rects[i][:2], rects[i][2:])
    # img = draw_rectangle(img, no_correction[i][:2], no_correction[i][2:])

    if i in [1, 100, 200, 300, 400]:
        # cv2.imwrite(f"../data/carseqrects{i}.jpg", img)
        cv2.imshow(f"{i}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

np.save("carseqrects-wcrt.npy", rects)
# i_last = img
