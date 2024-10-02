import cv2
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine, construct_M
from InverseCompositionAffine import InverseCompositionAffine
from LucasKanadeAffine import warp_image


def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance, iter_number=-1, do_inverse_compositional=False):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    if do_inverse_compositional:
        M = InverseCompositionAffine(image1, image2, threshold, num_iters, iter_number)
    else:
        p = LucasKanadeAffine(image1, image2, threshold, num_iters, iter_number)
        M = construct_M(p)

    warped_im1 = cv2.warpAffine(image1, M[:2, :], (image1.shape[1], image1.shape[0]))
    err_img = warped_im1 - image2

    mask = np.array(np.abs(err_img) > tolerance).astype(np.uint8)

    # if iter_number > 1:
    #     cv2.imshow("Warped Image 1", warped_im1)
    #     cv2.imshow("Image 2", image2)
    #     cv2.imshow("Motion", mask)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    return mask.astype(bool)
