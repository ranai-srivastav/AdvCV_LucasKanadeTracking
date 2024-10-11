import cv2
import numpy as np
from tqdm import tqdm
from pytools import delta
import numpy.typing as npt
from scipy.ndimage import shift
from scipy.interpolate import RectBivariateSpline


######################################### NOTES ###############################################
# OpenCV uses (x, y)
# loops use   (i, j)
# NumPy  uses (col, row)
#############################################################################################
def warp_image(img: npt.NDArray, warp_params: npt.NDArray) -> npt.NDArray:
    warped_coords =  warp_params @ img

    return warped_coords


def construct_M(p):
    M = np.array([[   (1+p[0]),        p[1],         p[2]],
                  [       p[3],    (1+p[4]),         p[5]],
                  [np.array([0]), np.array([0]), np.array([1])]
                 ])

    return M.squeeze(axis=2)

def calc_Hessian(x_grad_err, y_grad_err):
    flattened_x_grad = x_grad_err.flatten()
    flattened_y_grad = y_grad_err.flatten()

    dW_dp = np.eye(2)
    nabla_I = np.array([flattened_x_grad, flattened_y_grad]).T

    # A = dW_dp @ nabla_I
    H = nabla_I.T @ nabla_I

    return H


def calc_nabla_I(Ix, Iy) -> npt.NDArray:
    Ix = Ix.flatten().reshape(1, -1)
    Iy = Iy.flatten().reshape(1, -1)

    nabla_I = np.array([Ix, Iy])
    return nabla_I


def calc_jacobian(x_coords: npt.NDArray, y_coords: npt.NDArray, zeros, ones) -> npt.NDArray:
    dW_dp = np.array([[x_coords, y_coords, ones, zeros, zeros, zeros],
                      [zeros, zeros, zeros, x_coords, y_coords, ones]])

    return dW_dp.squeeze(axis=2)


def calc_deltaP(err_img, x_grad_err, y_grad_err, H):
    flattened_x_grad = x_grad_err.flatten()
    flattened_y_grad = y_grad_err.flatten()
    nabla_I = np.array([flattened_x_grad, flattened_y_grad]).T  # 2xN -> Nx2
    # 2x1 =               2x2 @  2xN @ Nx1
    delta_p = np.linalg.inv(H) @ (nabla_I.T @ err_img.flatten().reshape(-1, 1))

    return delta_p


def get_interpolated_img(rbs, x_coords, y_coords):
    # cv2.imshow("old_img", old_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    x_meshgrid, y_meshgrid = np.meshgrid(x_coords, y_coords)

    interpolated_img = rbs.ev(y_meshgrid, x_meshgrid)

    # cv2.imshow("interpolated images", interpolated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return interpolated_img


def LucasKanadeAffine(It, It1, threshold, num_iters, p0=np.zeros([6, 1])):
    """
    :param iter_number: Used for debugging only
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """

    # Calculating the X and Y derivatives of the image
    x_grad_It1 = cv2.Sobel(It1, cv2.CV_64F, 1, 0, ksize=3)
    y_grad_It1 = cv2.Sobel(It1, cv2.CV_64F, 0, 1, ksize=3)

    # Interpolation
    rbs_It = RectBivariateSpline(list(range(It.shape[0])), list(range(It.shape[1])), It)
    rbs_It1 = RectBivariateSpline(list(range(It1.shape[0])), list(range(It1.shape[1])), It1)
    rbs_gradX = RectBivariateSpline(list(range(x_grad_It1.shape[0])), list(range(x_grad_It1.shape[1])), x_grad_It1)
    rbs_gradY = RectBivariateSpline(list(range(y_grad_It1.shape[0])), list(range(y_grad_It1.shape[1])), y_grad_It1)

    num_pixels = It.shape[0] * It.shape[1]

    x_coords_It = np.arange(0, It.shape[1])    # increasing col numbers
    y_coords_It = np.arange(0, It.shape[0])    # increasing rows

    x_meshgrid, y_meshgrid = np.meshgrid(x_coords_It, y_coords_It)  # (cols, rows)
    x_meshgrid = x_meshgrid.flatten().reshape(1, -1)
    y_meshgrid = y_meshgrid.flatten().reshape(1, -1)

    n_ones = np.ones([1, num_pixels])
    n_zeros = np.zeros([1, num_pixels])

    dW_dp = calc_jacobian(x_meshgrid, y_meshgrid, n_zeros, n_ones)  # 2 x 6 x numpxl
    dW_dp = dW_dp.transpose([2, 0, 1])  # numpxl x 6 x 2

    It_flattened = It.flatten()
    It1_flattened = It1.flatten()

    homogenized_coords = np.vstack([x_meshgrid, y_meshgrid, n_ones])

    prev_delta_p = np.zeros([2, 1])
    p = p0.copy()

    for i in tqdm(range(num_iters)):

        M = construct_M(p) # 3 x 3
        warped_coordinates_It = warp_image(homogenized_coords, M)     # (x, y, 1) x numpxl  # row constant, col changes  # going row wise
        nohomo_coords = warped_coordinates_It[:2]/warped_coordinates_It[2]  # (row_idx, col_idx) x numpxl

        warped_valid_mask = np.array([
            [(nohomo_coords[0, :] >= 0) & (nohomo_coords[0, :] < It.shape[1])],
            [(nohomo_coords[1, :] >= 0) & (nohomo_coords[1, :] < It.shape[0])]
        ])
        warped_valid_mask = warped_valid_mask.squeeze(axis=1)    # (valid_row_idx, valid_col_idx) x numpxl

        warped_invalid_mask = ~warped_valid_mask
        nohomo_coords[warped_invalid_mask] = 0    #
        warped_x_coords = nohomo_coords[0]
        warped_y_coords = nohomo_coords[1]

        It1_x_warped = rbs_gradX.ev(warped_y_coords, warped_x_coords)
        It1_y_warped = rbs_gradY.ev(warped_y_coords, warped_x_coords)

        It1_x_warped[~warped_valid_mask[0]] = 0
        It1_y_warped[~warped_valid_mask[1]] = 0

        # if iter_number > 5:
        #     cv2.imshow("I XGrad", It1_x_warped.reshape([256, 256]))
        #     cv2.imshow("I YGrad", It1_y_warped.reshape([256, 256]))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        nabla_I = calc_nabla_I(It1_x_warped, It1_y_warped)   # (row, col) x 1 x numpxl
        nabla_I = nabla_I.transpose([2, 1, 0])   # numpxl x 1 x (row, col)

        # dW_dp is numpxl x 6 x (row, col)
        A_matmul = nabla_I @ dW_dp
        A_matmul = A_matmul.squeeze(axis=1)

        # Should I calculate it again?
        warped_It1 = rbs_It1.ev(warped_y_coords, warped_x_coords)

        # if iter_number > -1:
        #     cv2.imshow("Warped IT1", warped_It1.reshape([256, 256]))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        H = A_matmul.T @ A_matmul
        reconst_warped_It1 = warped_It1.reshape((It.shape[0], It.shape[1]))

        #TODO Set the err image at invalid masked coordinates to 0
        err_img = It - reconst_warped_It1

        delta_p = np.linalg.inv(H) @ (A_matmul.T @ err_img.flatten().reshape(-1, 1))

        p = p + delta_p

        if np.power(np.linalg.norm(delta_p), 2) < threshold:
            break

    return p


def draw_rectangle(img, top_left, bott_right, color=(0, 0, 255), thic=2):
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = (255 * img).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    top_left = [int(top_left[0]), int(top_left[1])]
    bott_right = [int(bott_right[0]), int(bott_right[1])]
    return cv2.rectangle(img, top_left, bott_right, color, thic)
