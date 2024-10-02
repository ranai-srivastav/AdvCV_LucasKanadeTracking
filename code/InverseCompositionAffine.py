import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
from tqdm import tqdm
import cv2


def get_jacobian_at_0(x_points, y_points, zeros, ones):
    x_points_flattened = x_points.flatten().reshape(1, 1, -1)
    y_points_flattened = y_points.flatten().reshape(1, 1, -1)
    ones = ones.reshape(1, 1, -1)
    zeros = zeros.reshape(1, 1, -1)

    dW_dp = np.array([[x_points_flattened, zeros, y_points_flattened, zeros, ones, zeros],
                      [zeros, x_points_flattened, zeros, y_points_flattened, zeros, ones]])
    # 2 x 6 x 1 x 1 x numpxl
    return dW_dp.squeeze().transpose(2, 0, 1)


def warp_image(img: NDArray, warp_params: NDArray) -> NDArray:
    warped_coords = warp_params @ img

    return warped_coords


def construct_M(p):
    M = np.array([[(1 + p[0]), p[2], p[4]],
                  [p[1], (1 + p[3]), p[5]],
                  [np.array([0]), np.array([0]), np.array([1])]
                  ])

    return M.squeeze(axis=2)


def get_inv_warp(p):
    det = 1 / ((1 + p[0]) * (1 + p[3]) - (p[1] * p[2]))

    new_p = np.array([[-p[0] - p[0] * p[3] + (p[1] * p[2])],
                      [-p[1]],
                      [-p[2]],
                      [-p[3] - p[0] * p[3] + p[1] * p[2]],
                      [-p[4] - p[3] * p[4] + p[2] * p[5]],
                      [-p[5] - p[0] * p[5] + p[1] * p[4]]
                     ])

    return (det * new_p).squeeze(axis=1)


def InverseCompositionAffine(It, It1, threshold, num_iters, iter_number=-1):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """
    # Template is It
    # Image is It1

    zeros = np.zeros(It.shape)
    ones = np.ones(It.shape)

    template_x_range = np.arange(0, It.shape[1])
    template_y_range = np.arange(0, It.shape[0])

    x_template_coords, y_template_coords = np.meshgrid(template_x_range, template_y_range)
    flat_template_x_coords = x_template_coords.flatten()
    flat_template_y_coords = y_template_coords.flatten()

    template_x_grad = cv2.Sobel(It, cv2.CV_64F, dx=1, dy=0, ksize=3)
    template_y_grad = cv2.Sobel(It, cv2.CV_64F, dx=0, dy=1, ksize=3)

    rbs_template = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    rbs_It1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)

    flat_template_x_grad = template_x_grad.flatten()
    flat_template_y_grad = template_y_grad.flatten()

    nabla_T = np.array([flat_template_x_grad, flat_template_y_grad])
    nabla_T = np.expand_dims(nabla_T, axis=1).T
    # nabla T = numpxl x 1 x (Tx, Ty)
    # dW_dp   = numpxl x (x, y) x 6
    dW_dp = get_jacobian_at_0(flat_template_x_coords, flat_template_y_coords, zeros, ones)

    homogenized_image_coords = np.array([ [flat_template_x_coords], [flat_template_y_coords], [np.ones(flat_template_x_coords.shape)]])
    homogenized_image_coords = homogenized_image_coords.transpose([2, 0, 1])

    # A = (numpxl x 1 x 2) @ (numpxl x 2 x 6)
    A = nabla_T @ dW_dp
    A = A.squeeze()  # numpxl x 6
    H = A.T @ A  # (6 x numpxl) @ (numpxl x 6) = 6x6

    p = np.zeros((6, 1))
    M = construct_M(p)

    for i in tqdm(range(num_iters)):
        warped_homogenized_image_coords = warp_image(homogenized_image_coords, M)
        warped_homogenized_image_coords = warped_homogenized_image_coords.transpose([1, 2, 0])
        warped_dehomogenized_image_coords = warped_homogenized_image_coords[:2] / warped_homogenized_image_coords[2]

        warped_image_intensities = rbs_It1.ev(warped_dehomogenized_image_coords[1],
                                              warped_dehomogenized_image_coords[0])

        # if iter_number > -1:
        #     cv2.imshow("Image after warping", warped_image_intensities.reshape(It.shape))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        # Template - Warped Image
        warped_dehomogenized_image_x_coords = warped_dehomogenized_image_coords[0]
        warped_dehomogenized_image_y_coords = warped_dehomogenized_image_coords[1]

        mask_x = np.array(
            [(warped_dehomogenized_image_x_coords >= 0) & (warped_dehomogenized_image_x_coords < It.shape[1])])
        mask_y = np.array(
            [(warped_dehomogenized_image_y_coords >= 0) & (warped_dehomogenized_image_y_coords < It.shape[0])])

        valid_warped_img_points_mask = np.array(mask_x & mask_y)
        invalid_warped_img_points_mask = ~valid_warped_img_points_mask
        invalid_warped_img_points_mask = invalid_warped_img_points_mask.squeeze(axis=1)

        warped_image_intensities[invalid_warped_img_points_mask] = 0

        warped_image_reconst = warped_image_intensities.reshape(It.shape)

        # if iter_number > -1:
        #     cv2.imshow("Image after warping", warped_image_reconst.reshape(It.shape))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        err_img = warped_image_reconst - It

        # if iter_number > -1:
        #     cv2.imshow("Err Img", err_img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        delta_p = np.linalg.inv(H) @ A.T @ err_img.flatten()

        delta_p_inv = get_inv_warp(p)
        M_inv = construct_M(delta_p_inv)
        M = M @ M_inv

        if np.power(np.linalg.norm(delta_p), 2) < threshold:
            break

    return M[:2, :]
