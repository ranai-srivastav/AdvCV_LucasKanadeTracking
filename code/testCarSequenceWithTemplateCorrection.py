from tempfile import template

import cv2
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from LucasKanade import LucasKanade, draw_rectangle, add_p_to_rect
# from LucasKanadeFlattened import LucasKanadeFlattened, draw_rectangle, add_p_to_rect
from LucasKanadeRBS import LucasKanadeRBS, draw_rectangle, add_p_to_rect

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

rect = [59, 116, 145, 151]
rect0 = rect.copy()
# rects = np.zeros([seq.shape[2], 4])
# rects[0] = rect.copy()

prev_p = np.zeros([2, 1])
prev_rect = np.zeros([1, 4])

I0 = seq[:, :, 0]

# for i in tqdm(range(0, np.shape(seq)[2] - 1)):
for i in range(0, np.shape(seq)[2] - 1):
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]

    # if i == 194:
    #     cv2.imshow("It", It)
    #     cv2.imshow("It", It1)

    # rect is updated with values from the prev iteration
    # rect diff from t=0 to now
    p_diff = np.array([rect[0] - rect0[0], rect[1] - rect0[1]]).reshape(2, 1)

    p_n = LucasKanadeRBS(It, It1, rect, threshold, int(num_iters), i)
    p_n_starred = p_n + p_diff
    p_star = LucasKanadeRBS(I0, It1, rect0, threshold, int(num_iters), iter_number=i, p0=p_n_starred.copy())

    if (np.linalg.norm(p_star - p_n_starred)) < template_threshold:
        prev_p = p_star.copy()
        rect = add_p_to_rect(rect0, p_star)
    else:
        prev_p = p_n.copy()
        rect = add_p_to_rect(rect, p_n)

    # rects[i, :] = np.array(rect).reshape(1, 4)

    img = draw_rectangle(It1, rect[:2], rect[2:])

    # if i in [1, 10, 20, 40, 80, 100, 150, 160, 170, 180, 190, 200,250, 300, 350, 400, 414]:
    if i in [12, 13, 14, 15, 16, 17, 50, 100, 150, 200, 250, 300, 400]:
        cv2.imwrite(f"../data/carseqrects{i}.jpg", img)
        cv2.imshow(f"{i}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# np.save("carseqrects-wcrt.npy", rects)
# i_last = img
