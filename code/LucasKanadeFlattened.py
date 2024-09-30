import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import shift
from scipy.interpolate import RectBivariateSpline

######################################### NOTES ###############################################
# OpenCV uses (x, y)
# loops use   (i, j)
# NumPy  uses (col, row)
#############################################################################################
def add_p_to_rect(rect_coords, p):
    return [rect_coords[0] + p[0], rect_coords[1] + p[1], rect_coords[2] + p[0], rect_coords[3] + p[1]]


def calc_Hessian(x_grad_err, y_grad_err):
    flattened_x_grad = x_grad_err.flatten()
    flattened_y_grad = y_grad_err.flatten()

    dW_dp = np.eye(2)
    nabla_I = np.array([flattened_x_grad, flattened_y_grad]).reshape(-1, 2)
    # A = dW_dp @ nabla_I

    H = nabla_I.T @ nabla_I

    return H


def calc_deltaP(err_img, x_grad_err, y_grad_err, H):

    flattened_x_grad = x_grad_err.flatten()
    flattened_y_grad = y_grad_err.flatten()
    nabla_I = np.array([flattened_x_grad, flattened_y_grad]).reshape(-1, 2) # Nx2
    # 2x1 =               2x2 @  2xN @ Nx1
    delta_p = np.linalg.inv(H) @ (nabla_I.T @ err_img.flatten().reshape(-1, 1))

    return delta_p

def get_interpolated_img(old_img, old_top_left, old_bott_right, translation):

    # cv2.imshow("old_img", old_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    rbs = RectBivariateSpline(list(range(old_img.shape[0])), list(range(old_img.shape[1])), old_img)

    x_coords = np.arange(old_top_left[0], old_bott_right[0], 1) + translation[0]
    y_coords = np.arange(old_top_left[1], old_bott_right[1], 1) + translation[1]

    x_meshgrid, y_meshgrid = np.meshgrid(x_coords, y_coords)

    interpolated_img = rbs.ev(y_meshgrid, x_meshgrid)
    # cv2.imshow("interpd_img", interpolated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return interpolated_img


def LucasKanadeFlattened(It, It1, rect, threshold, num_iters, iter_number=-1, p0=np.zeros([2, 1])):
    """
    :param iter_number:
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    top_left_t = np.array([rect[0], rect[1]+1])
    bottom_right_t = np.array([rect[2], rect[3]+1])

    template_t = get_interpolated_img(It, top_left_t, bottom_right_t, np.zeros((2, 1)))

    if iter_number == 20:
        cv2.imshow("Template", template_t)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    prev_delta_p = np.zeros([2, 1])
    p = p0.copy()

    for i in range(num_iters):

        warped_template = get_interpolated_img(It1, top_left_t, bottom_right_t, p)
        if iter_number == 20:
            cv2.imshow("Warped", warped_template)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # TODO Change order
        err_img = template_t - warped_template
        # err_img = warped_template - template_t

        x_grad_warped_template = cv2.Sobel(warped_template, cv2.CV_64F, 1, 0, ksize=3)
        y_grad_warped_template = cv2.Sobel(warped_template, cv2.CV_64F, 0, 1, ksize=3)

        # cv2.imshow("err_img", err_img)
        # # cv2.imshow("XSobel", x_grad_warped_template)
        # # cv2.imshow("YSobel", y_grad_warped_template)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        H = calc_Hessian(x_grad_warped_template, y_grad_warped_template)
        delta_p = calc_deltaP(err_img, x_grad_warped_template, y_grad_warped_template, H)

        p += delta_p

        if np.all(np.abs(delta_p-prev_delta_p) < threshold):
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
