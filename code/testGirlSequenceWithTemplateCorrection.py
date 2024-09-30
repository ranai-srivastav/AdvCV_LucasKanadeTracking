import cv2
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanadeRBS import LucasKanadeRBS, draw_rectangle, add_p_to_rect

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=0.00001,
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

seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

rect = np.array(rect).reshape(1, 4)
rects = np.zeros( [seq.shape[2], 4])
rects[0] = rect
seq = 255 - seq
I0 = seq[:, :, 0]

for i in range(seq.shape[2] - 1):
    It = seq[:, :, i]
    It1 = seq[:, :, i + 1]

    p_diff = np.array([rects[i][0] - rects[0][0], rects[i][1] - rects[0][1]]).reshape(2, 1)

    p_n = LucasKanadeRBS(It, It1, rects[i, :], threshold, int(num_iters), i)
    p_n_starred = p_n + p_diff
    p_star = LucasKanadeRBS(I0, It1, rects[i, :], threshold, int(num_iters), i, p0=p_n_starred)

    if np.linalg.norm(p_star - p_n_starred) < template_threshold:
        print(f"{i}: chose pstar")
        prev_p = p_star.copy()
        rects[i+1, :] = np.array(add_p_to_rect(rects[0], p_star)).reshape(1, 4)
    else:
        print(f"{i}: chose pn")
        prev_p = p_n.copy()
        rects[i+1, :] = np.array(add_p_to_rect(rects[i], p_n)).reshape(1, 4)

    img = draw_rectangle(It1, rects[i+1][:2], rects[i+1][2:])

    if i in [1, 20, 40, 60, 80]:
        cv2.imwrite(f"../data/girlseqrects_wcrt{i}.jpg", img)
        cv2.imshow(f"{i}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

np.save("girlseqrects_wcrt.npy", rects)





