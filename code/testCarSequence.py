import cv2
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from LucasKanade import LucasKanade, draw_rectangle, add_p_to_rect
# from LucasKanadeFlattened import LucasKanadeFlattened, draw_rectangle, add_p_to_rect
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
rects[0] = rect.copy()

# cv2.imshow("LKT", draw_rectangle(seq[:, :, 0], rect[:2], rect[2:]))
# cv2.waitKey(0)

im1 = im100 = im200 = im300 = im400 = np.zeros(seq.shape[:2])

for i in tqdm(range(0, np.shape(seq)[2] - 1)):
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]

    # cv2.imshow("It", It)
    # cv2.imshow("It1", It1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    delta_p = LucasKanadeRBS(It, It1, rect, threshold, int(num_iters), i)
    # print(delta_p)
    rect = add_p_to_rect(rect, delta_p)
    img = draw_rectangle(It1, rect[:2], rect[2:])
    rects[i, :] = np.array(rect).reshape(1, 4)

    if i in [1, 100, 200, 300, 400]:
        cv2.imshow(f"Frame {i}", img)
        # cv2.imwrite(f"../data/CarSeq_{i}.jpg", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# np.save("CarSeq.npy", rects)

# cv2.imshow("Frame 1", im1)
# cv2.imshow("Frame 100", im100)
# cv2.imshow("Frame 200", im200)
# cv2.imshow("Frame 300", im300)
# cv2.imshow("Frame 400", im400)
# cv2.imshow("Last Frame", i_last)
# cv2.waitKey(0)
cv2.destroyAllWindows()
