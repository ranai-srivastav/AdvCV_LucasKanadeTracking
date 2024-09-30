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


def calc_Hessian(err_img, x_grad_err, y_grad_err):
    H = np.zeros([2, 2])

    for row_idx in range(len(err_img)):
        for col_idx in range(len(err_img[row_idx])):

            nabla_I = np.array( [[x_grad_err[row_idx, col_idx], y_grad_err[row_idx, col_idx]]] )
            H += nabla_I.T @ nabla_I

    return H


def calc_deltaP(err_img, x_grad_err, y_grad_err, H):
    delta_p = np.array([0, 0]).reshape(2, 1).astype(np.float64)
    for row_idx in range(len(err_img)):
        for col_idx in range(len(err_img[row_idx])):
            nabla_I = np.array([[x_grad_err[row_idx, col_idx], y_grad_err[row_idx, col_idx]]]) # 1x2
            # TODO HACKY - Why do I need to do np.array() around the error image
            delta_p += nabla_I.T @ np.array([err_img[row_idx, col_idx]]).reshape(1, 1)

    delta_p = np.linalg.inv(H) @ delta_p

    return delta_p


def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros([2, 1])):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    top_left_t = (rect[0], rect[1]+1)
    bottom_right_t = (rect[2], rect[3]+1)

    template_t = It[top_left_t[1]:bottom_right_t[1], top_left_t[0]:bottom_right_t[0]]

    rbs = RectBivariateSpline(list(range(It1.shape[0])), list(range(It1.shape[1])), It1)

    prev_delta_p = np.zeros(2)
    p = p0.copy()

    for i in tqdm(range(num_iters)):

        warped_template = np.zeros(template_t.shape)
        err_img = np.zeros(template_t.shape)

        for row_idx in range(template_t.shape[0]):   # iterating through rows
            for col_idx in range(template_t.shape[1]):
                warped_template[row_idx, col_idx] = rbs.ev(row_idx+p[0], col_idx+p[1])

        err_img = template_t - warped_template

        x_grad_err_img = cv2.Sobel(err_img, cv2.CV_64F, 1, 0, ksize=3)
        y_grad_err_img = cv2.Sobel(err_img, cv2.CV_64F, 0, 1, ksize=3)

        # cv2.imshow("err_img", err_img)
        # cv2.imshow("XSobel", x_grad_err_img)
        # cv2.imshow("YSobel", y_grad_err_img)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()

        H = calc_Hessian(err_img, x_grad_err_img, y_grad_err_img)
        delta_p = calc_deltaP(err_img, x_grad_err_img, y_grad_err_img, H)

        p += delta_p

        if np.all(prev_delta_p - delta_p < threshold):
            break

    return p


def draw_rectangle(img, top_left, bott_right, color=(0, 0, 255), thic=2):

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = (255*img).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    top_left = [int(top_left[0]), int(top_left[1])]
    bott_right = [int(bott_right[0]), int(bott_right[1])]
    return cv2.rectangle(img, top_left, bott_right, color, thic)
