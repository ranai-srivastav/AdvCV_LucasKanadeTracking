import cv2
import numpy as np
from pytools import delta
from tqdm import tqdm
from scipy.ndimage import shift
from scipy.interpolate import RectBivariateSpline


######################################### NOTES ###############################################
# OpenCV uses (x, y)
# loops use   (i, j)
# NumPy  uses (col, row)
#############################################################################################
def add_p_to_rect(rect_coords, p):
    return [rect_coords[0] + p[0], rect_coords[1] + p[1],
            rect_coords[2] + p[0], rect_coords[3] + p[1]]

def calc_Hessian(x_grad_err, y_grad_err):
    flattened_x_grad = x_grad_err.flatten()
    flattened_y_grad = y_grad_err.flatten()

    dW_dp = np.eye(2)
    nabla_I = np.array([flattened_x_grad, flattened_y_grad]).T

    # A = dW_dp @ nabla_I
    H = nabla_I.T @ nabla_I

    return H


def calc_deltaP(err_img, x_grad_err, y_grad_err, H):

    flattened_x_grad = x_grad_err.flatten()
    flattened_y_grad = y_grad_err.flatten()
    nabla_I = np.array([flattened_x_grad, flattened_y_grad]).T # 2xN -> Nx2
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


def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros([2, 1])):
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
    rbs_It =    RectBivariateSpline(list(range(It.shape[0])),         list(range(It.shape[1])), It)
    rbs_It1 =   RectBivariateSpline(list(range(It1.shape[0])),        list(range(It1.shape[1])), It1)
    rbs_gradX = RectBivariateSpline(list(range(x_grad_It1.shape[0])), list(range(x_grad_It1.shape[1])), x_grad_It1)
    rbs_gradY = RectBivariateSpline(list(range(y_grad_It1.shape[0])), list(range(y_grad_It1.shape[1])), y_grad_It1)

    top_left_x, top_left_y = rect[0], rect[1]
    bott_right_x, bott_right_y = rect[2]+1, rect[3]+1

    # DEBUG
    # if iter_number == 20:
        # cv2.imshow("Template", template_t)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    prev_delta_p = np.zeros([2, 1])
    p = p0.copy()

    for i in range(num_iters):

        x_coords = np.arange(top_left_x, bott_right_x + 1)
        y_coords = np.arange(top_left_y, bott_right_y + 1)
        warped_x_coords = x_coords + p[0]
        warped_y_coords = y_coords + p[1]

        template_t = get_interpolated_img(rbs_It, x_coords, y_coords)

        warped_rect = add_p_to_rect(rect, p)
        # Should be 36x87
        warped_t1 = get_interpolated_img(rbs_It1, warped_x_coords, warped_y_coords)

        x_grad_warped_template = get_interpolated_img(rbs_gradX, warped_x_coords, warped_y_coords)
        y_grad_warped_template = get_interpolated_img(rbs_gradY, warped_x_coords, warped_y_coords)

        err_img = template_t - warped_t1
        # err_img = warped_t1 - template_t

        H = calc_Hessian(x_grad_warped_template, y_grad_warped_template)
        delta_p = calc_deltaP(err_img, x_grad_warped_template, y_grad_warped_template, H)

        p += delta_p

        # if np.all(np.abs(delta_p - prev_delta_p)) < threshold:
        # if np.all(delta_p) < threshold:
        if np.power(np.linalg.norm(delta_p), 2) < threshold:
            break

        prev_delta_p = delta_p

    return p


def draw_rectangle(img, top_left, bott_right, color=(0, 0, 255), thic=2):

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = (255*img).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    top_left = [int(top_left[0]), int(top_left[1])]
    bott_right = [int(bott_right[0]), int(bott_right[1])]
    return cv2.rectangle(img, top_left, bott_right, color, thic)
