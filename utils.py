import cv2
import base64


def read_image_as_arr(img_path):
    img = cv2.imread(img_path)
    return img


def read_image_as_str(img_path):
    with open(img_path, "rb") as image:
        image_str = base64.b64encode(image.read()).decode("utf-8")
    return image_str
