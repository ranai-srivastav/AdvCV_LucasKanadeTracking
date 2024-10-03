import cv2
import argparse
import numpy as np
from tqdm import tqdm

from LucasKanadeRBS import LucasKanadeRBS, draw_rectangle, add_p_to_rect


parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=0.001,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")

rect = [59, 116, 145, 151]
rects = np.zeros([seq.shape[2], 4])
rects[0] = np.array([rect]).reshape([1, 4])

im1 = im100 = im200 = im300 = im400 = np.zeros(seq.shape[:2])

for i in tqdm(range(1, np.shape(seq)[2])):
    It = seq[:, :, i-1]
    It1 = seq[:, :, i]

    delta_p = LucasKanadeRBS(It, It1, rects[i-1], threshold, int(num_iters), i)
    rects[i] = np.array(add_p_to_rect(rects[i-1], delta_p)).reshape(1, 4)
    img = draw_rectangle(It1, rects[i][:2], rects[i][2:])

    if i in [1, 100, 200, 300, 400]:
        cv2.imshow(f"Frame {i}", img)
        cv2.imwrite(f"../data/CarSeq_{i}.jpg", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

np.save("carseqrects.npy", rects)
cv2.destroyAllWindows()
