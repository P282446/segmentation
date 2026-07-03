import cv2
import math
import numpy as np
from PIL import Image
from scipy.signal import convolve2d


def open_file(filename):
    return duoImage(Image.open(filename))


class duoImage:
    """
    Maintains in parallel two representations of the image : a pillow version and an opencv version (stored as a bitmap numpy array).
    Not comitting to a unique representation allows us to make conversions only when necessary
    """

    def __init__(self, image1):

        self.pil_is_up_to_date = False
        self.opencv_is_up_to_date = False
        self.pil_version = None
        self.opencv_version = None
        if isinstance(image1, duoImage):
            self.pil_is_up_to_date = image1.pil_is_up_to_date
            self.opencv_is_up_to_date = image1.opencv_is_up_to_date
            self.opencv_version = image1.opencv_version
            self.pil_version = image1.pil_version
        elif isinstance(image1, Image.Image):
            self.pil_version = image1
            self.pil_is_up_to_date = True
        elif isinstance(image1, np.ndarray):
            self.opencv_version = image1
            self.opencv_is_up_to_date = True
        else:
            raise Exception("Invalid image type")

    def copy(self):
        new_duo_image = duoImage(self)
        return new_duo_image

    @property
    def pil_version(self):
        if self._pil_is_up_to_date:
            return self._pil_version
        else:
            self._pil_version = image_to_pil(self.opencv_version)
            self._pil_is_up_to_date = True
            return self._pil_version

    @pil_version.setter
    def pil_version(self, value):
        self._pil_version = value
        self._opencv_is_up_to_date = False
        self._pil_is_up_to_date = True

    @property
    def opencv_version(self):
        if self._opencv_is_up_to_date:
            return self._opencv_version
        else:
            self._opencv_version = pil_to_image(self.pil_version)
            self._opencv_is_up_to_date = True
            return self._opencv_version

    @opencv_version.setter
    def opencv_version(self, value):
        self._opencv_version = value
        self._pil_is_up_to_date = False
        self._opencv_is_up_to_date = True

    @property
    def size(self):
        if self._pil_is_up_to_date:
            return self._pil_version.size
        elif self._opencv_is_up_to_date:
            h, w, _ = self._opencv_version.shape
            return w, h
        raise Exception("Not up to date")


def build_base_converting_function(x, y):
    """
    Build a translation converting function
    """

    def convert_coordinates(cx, cy):
        cx_original = cx + x
        cy_original = cy + y
        return cx_original, cy_original

    return convert_coordinates


def build_rotation_converting_function(x1, y1, x2, y2, angle_radian):
    "Build a rotation converting function"
    cos = math.cos(angle_radian)
    sin = math.sin(angle_radian)

    def convert_coordinates(cx, cy):
        return x2 + (cx - x1) * cos + (cy - y1) * sin, y2 - (cx - x1) * sin + (
            cy - y1) * cos

    return convert_coordinates


def correct_uneven_lighting(image, lpf_kernel_size=(100, 100)):
    """
    Corrects uneven lighting modeling lighting affects as low frequency components of the image
    """
    lpf = cv2.blur(image.opencv_version, lpf_kernel_size)
    mean_lpf = np.mean(lpf)

    corrected_image = image.opencv_version - 0.8 * lpf + 0.8 * mean_lpf
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)

    return duoImage(corrected_image)


def image_to_pil(img):
    """
    Convert image from opencv to pillow formats
    """
    cv2_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_image_rgb)


def pil_to_image(image_pil):
    """
    Convert image from pillow to opencv formats
    """
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_BGR2RGB)


def crop_center(image, new_width, new_height):
    """
    Crops image around center.
    """
    width, height = image.size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return duoImage(image.pil_version.crop((left, top, right, bottom)))


def rotate_image(image, degree):
    """
    Rotates image around center, expanding the frame to include all elements.

    Resampling is necessary as rotation introduces artifacts in the image.
    """
    return duoImage(
        image.pil_version.rotate(degree,
                                 fillcolor=(255, 255, 255),
                                 expand=True,
                                 resample=Image.BICUBIC))


def rotate_image_without_expanding(image, degree):
    """
    Rotates image around center, maitining the frame's size.

    Resampling is necessary as rotation introduces artifacts in the image.
    """
    return duoImage(
        image.pil_version.rotate(degree,
                                 fillcolor=(255, 255, 255),
                                 expand=False,
                                 resample=Image.BICUBIC))


def new_dimensions(w, h, angle):
    image = rotate_image(duoImage(np.zeros((w, h, 3), dtype=np.uint8)), angle)
    return image.size


def alpha_blend(image1, image2, alpha):
    blended_image = cv2.addWeighted(image1.opencv_version, 1,
                                    image2.opencv_version, alpha, 0)
    return duoImage(blended_image)


def print_all_zones_on_image(img, landmarks, zones):
    img_ = img.copy()
    if len(landmarks) > 0:
        alphas = lambda n: 0.05 + (1 - 1 / (n + 1)) * 0.1
        colors = lambda n: (int(255 * math.cos(
            n / 2)), int(255 * math.sin(n / 2)), int(255 * math.cos(n**2 / 4)))
        for i, zone in enumerate(zones):

            mask = np.zeros_like(img_.opencv_version)
            for j in range(len(zone)):
                point1 = landmarks[zone[j]]
                point2 = landmarks[zone[(j + 1) % len(zone)]]
                cv2.line(mask, point1, point2, (255, 255, 0), 1)
            img_ = duoImage(alpha_blend(img_, duoImage(mask), 1))
            mask = np.zeros_like(img_.opencv_version)
            if len(zone) >= 3:
                selected_landmarks = np.array([landmarks[s] for s in zone])
                closed_loop = np.vstack(
                    (selected_landmarks, selected_landmarks[0]))
                cv2.fillPoly(mask, [closed_loop.astype(np.int32)], colors(i))

            img_ = duoImage(alpha_blend(img_, duoImage(mask), alphas(i)))
        return img_
    return img


def resize_image(img, target_size=1500):

    width, height = img.size
    if width >= target_size and height >= target_size:
        return img.copy()
    aspect_ratio = width / height if width > height else height / width

    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size / aspect_ratio)

    resized_img = duoImage(
        img.pil_version.resize((new_width, new_height), Image.LANCZOS))

    return resized_img


def sharpen_image(image):
    image_array = np.array(image.opencv_version)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image_array, -1, kernel)
    return duoImage(sharpened)


def print_all_landmarks_on_image(img, landmarks, size=3):
    img_ = img.opencv_version
    for landmark in landmarks:
        cv2.circle(img_, landmark, size, (0, 0, 255), -1)
    return duoImage(img_)
