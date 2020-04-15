import numpy as np
import base64
import sys
import cv2
import time
import settings
import skimage.color
import skimage.io
import skimage.transform
from distutils.version import LooseVersion


def base64_encode_image(image_array):
    # base64 encode the input NumPy array
    return base64.b64encode(image_array).decode("utf-8")


def base64_decode_image(image_array):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        image_array = bytes(image_array, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    return base64.decodebytes(image_array)


def smallest_size_at_least(height, width, resize_min):
    smaller_dim = min(height, width)
    scale_ratio = resize_min / smaller_dim
    new_height = int(height * scale_ratio)
    new_width = int(width * scale_ratio)
    return new_height, new_width


def resize_image(image, height, width):
    return cv2.resize(image, (width, height))


def aspect_preserving_resize(image, resize_min):
    height, width = image.shape[0], image.shape[1]
    new_height, new_width = smallest_size_at_least(height, width, resize_min)
    return resize_image(image, new_height, new_width)


def central_crop(image, crop_height, crop_width):
    height, width = image.shape[0], image.shape[1]
    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return image[crop_top:crop_top + crop_height, crop_left:crop_left + crop_width]


def normalization(image, means):
    """
    Normalization with given means
    :param image: OpenCV Mat
    :param means: [R, G, B] ,e.g, [103.939, 116.779, 123.68]
    :return: OpenCV Mat
    """
    image = image.astype(np.float)
    image[:, :, 0] -= means[0]
    image[:, :, 1] -= means[1]
    image[:, :, 2] -= means[2]
    return image


def byte_to_mat(image_bytes, dtype):
    if isinstance(image_bytes, str):
        image_array = np.fromstring(image_bytes, dtype=dtype)
    else:
        image_array = np.frombuffer(image_bytes, dtype=dtype)
    # cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return cv2.imdecode(image_array, -1)


def NHWC2HCHW(image):
    # NHWC -> NCWH
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image


def image_preprocess(image, output_width, output_height):
    image = aspect_preserving_resize(image, settings.RESIZE_MIN)
    image = central_crop(image, output_height, output_width)
    # image = normalization(image, [103.939, 116.779, 123.68])
    image = NHWC2HCHW(image)
    # ensure our NumPy array is C-contiguous as well,
    # otherwise we won't be able to serialize it
    image = image.copy(order="C")
    return image


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    # Resizes an image keeping the aspect ratio unchanged
    image_dtype = image.dtype
    h, w = image.shape[:2]
    scale = 1

    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

        # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

        # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

        # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
    return image.astype(image_dtype)


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)