import numpy as np
import cv2


def compute_difference(bg_img, input_img):
    difference = cv2.absdiff(bg_img, input_img)

    return difference


def compute_binary_mask(difference_single_channel):
    _, difference_binary = cv2.threshold(src=difference_single_channel,
                                         thresh=15,
                                         maxval=225,
                                         type=cv2.THRESH_BINARY)
    return difference_binary


def replace_background(bg1_image, bg2_image, ob_image):
    difference_single_channel = compute_difference(
        bg1_image,
        ob_image
    )
    binary_mask = compute_binary_mask(difference_single_channel)
    output = np.where(binary_mask == 0, bg2_image, ob_image)
    return output


def run():
    bg1_image = cv2.imread("GreenBackground.png", 1)
    bg1_image = cv2.resize(bg1_image, (678, 381))

    ob_image = cv2.imread("Object.png", 1)
    ob_image = cv2.resize(ob_image, (678, 381))

    bg2_image = cv2.imread("NewBackground.jpg", 1)
    bg2_image = cv2.resize(bg2_image, (678, 381))

    window_name = 'image'
    difference_single_channel = replace_background(bg1_image, bg2_image, ob_image)
    cv2.imshow(window_name, difference_single_channel)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()


run()
