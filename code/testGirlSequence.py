import cv2
import argparse
import numpy as np
from tqdm import tqdm
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
    default=1e-1,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

# To make the image not inverted
seq = 255 - seq

for i in tqdm(range(0, np.shape(seq)[2] - 1)):
    It = seq[:, :, i]
    It1 = seq[:, :, i + 1]

    # cv2.imshow("It", It)
    # cv2.imshow("It1", It1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    delta_p = LucasKanadeRBS(It, It1, rect, threshold, int(num_iters), i)
    # print(delta_p)
    rect = add_p_to_rect(rect, delta_p)
    img = draw_rectangle(It1, rect[:2], rect[2:])

    if i in [1, 10, 20, 40, 80, 100, 200, 400, 414]:
        cv2.imshow(f"Frame {i}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
i_last = img

# cv2.imshow("Frame 1", im1)
# cv2.imshow("Frame 100", im100)
# cv2.imshow("Frame 200", im200)
# cv2.imshow("Frame 300", im300)
# cv2.imshow("Frame 400", im400)
# cv2.imshow("Last Frame", i_last)
# cv2.waitKey(0)
cv2.destroyAllWindows()
