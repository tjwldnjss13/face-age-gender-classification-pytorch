import cv2 as cv
import numpy as np

from PIL import Image


def shift_random(image):
    d = np.random.randint(0, 4)
    if d == 0:
        return shift_left(image)
    if d == 1:
        return shift_right(image)
    if d == 2:
        return shift_up(image)
    if d == 3:
        return shift_down(image)


# def upsample_dataset(data_list, num_target):
#     """
#     :param data_list: list of PIL Image
#     :param num_target: int, the brief number of augmented image
#
#     :return: list of augmented PIL Image
#     """
#     sampled_data_list = []
#     t = 0
#     d = 0
#     while len(data_list) + len(sampled_data_list) < num_target:
#         if t % 4 == 0:
#             d += 1
#             for data in data_list:
#                 sampled_data_list.append(shift_left(data, d))
#         elif t % 4 == 1:
#             for data in data_list:
#                 sampled_data_list.append(shift_right(data, d))
#         elif t % 4 == 2:
#             for data in data_list:
#                 sampled_data_list.append(shift_up(data, d))
#         else:
#             for data in data_list:
#                 sampled_data_list.append(shift_down(data, d))
#         t += 1
#
#     return data_list + sampled_data_list


def shift_left(image, distance=0):
    """
    :param image: PIL Image
    :param distance: int, amount of shift translation (pixel)

    :return: Augmented PIL Image
    """
    if distance == 0:
        distance = np.random.randint(0, 10)
    M = np.float32([[1, 0, -distance], [0, 1, 0]])

    img = np.array(image)
    H, W = img.shape[:2]
    img_shift = cv.warpAffine(img, M, (W, H))
    img_shift = Image.fromarray(img_shift)

    return img_shift


def shift_right(image, distance=0):
    """
    :param image: PIL Image
    :param distance: int, amount of shift translation (pixel)

    :return: Augmented PIL Image
    """
    if distance == 0:
        distance = np.random.randint(0, 10)
    M = np.float32([[1, 0, distance], [0, 1, 0]])

    img = np.array(image)
    H, W = img.shape[:2]
    img_shift = cv.warpAffine(img, M, (W, H))
    img_shift = Image.fromarray(img_shift)

    return img_shift


def shift_up(image, distance=0):
    """
    :param image: PIL Image
    :param distance: int, amount of shift translation (pixel)

    :return: Augmented PIL Image
    """
    if distance == 0:
        distance = np.random.randint(0, 10)
    M = np.float32([[1, 0, 0], [0, 1, -distance]])

    img = np.array(image)
    H, W = img.shape[:2]
    img_shift = cv.warpAffine(img, M, (W, H))
    img_shift = Image.fromarray(img_shift)

    return img_shift


def shift_down(image, distance=0):
    """
    :param image: PIL Image
    :param distance: int, amount of shift translation (pixel)

    :return: Augmented PIL Image
    """
    if distance == 0:
        distance = np.random.randint(0, 10)
    M = np.float32([[1, 0, 0], [0, 1, distance]])

    img = np.array(image)
    H, W = img.shape[:2]
    img_shift = cv.warpAffine(img, M, (W, H))
    img_shift = Image.fromarray(img_shift)

    return img_shift


if __name__ == '__main__':
    img_pth = '../samples/2007_001430.jpg'
    img = Image.open(img_pth)
    img_shift = shift_random(img)
    img_shift = np.array(img_shift)
    cv.imshow('image', img_shift)
    if cv.waitKey(0) == ord('q'):
        cv.destroyAllWindows()
















