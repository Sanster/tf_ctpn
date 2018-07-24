import cv2


def read_rgb_img(img_file_path):
    bgr = cv2.imread(img_file_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb
