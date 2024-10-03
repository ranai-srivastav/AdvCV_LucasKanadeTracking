import cv2
import argparse
import numpy as np
from tqdm import tqdm
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
rects = np.zeros([seq.shape[2], 4])
rect = [280, 152, 330, 318]
rects[0, :] = np.array(rect).reshape([1, 4])

# To make the image not inverted
seq = 255 - seq

for i in tqdm(range(0, np.shape(seq)[2] - 1)):
    It = seq[:, :, i]
    It1 = seq[:, :, i + 1]

    delta_p = LucasKanadeRBS(It, It1, rects[i], threshold, int(num_iters), i)
    # print(delta_p)
    rects[i+1, :] = np.array(add_p_to_rect(rects[i], delta_p)).reshape([1, 4])
    img = draw_rectangle(It1, rects[i][:2], rects[i][2:])

    if i in [1, 20, 40, 60, 80]:
        cv2.imshow(f"Frame {i}", img)
        cv2.imwrite(f"../data/girlseq_{i}.jpg", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

np.save("girlseq.npy", rects)
cv2.destroyAllWindows()
