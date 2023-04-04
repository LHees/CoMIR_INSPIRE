import random
import six.moves as sm

import numpy as np
from scipy import ndimage


class Augmentor():
    def __init__(self):
        self.crop_mode = None
        self.crop_size = None
        self.blur_sigma = None
        self.blur_p = 0
        self.fliplr = False
        self.fliplr_p = 0

    def augment(self, image: np.ndarray):  # augment the image in situ
        z, y, x, c = image.shape  # assuming z*y*x*c shape

        if self.crop_mode is not None and self.crop_size is not None:  # crop
            assert all(self.crop_size <= dim for dim in [z, y, x]), \
                f"The image is too small for the requested crop! Image size:" \
                f" {z}*{y}*{x} (z*y*x). Crop size: {self.crop_size}"

            crop_point = None
            if self.crop_mode == 'uniform':
                sample_space = self._compute_crop_sample_space(z, y, x)
                crop_point = [random.choice(r) for r in sample_space]
            elif self.crop_mode == 'center':
                crop_point = [z//2, y//2, x//2]
            crop_slice = tuple([slice(point - self.crop_size//2,
                                      point + self.crop_size//2)
                                for point in crop_point])
            image = image[crop_slice]

        if self.blur_sigma is not None and random.random() <= self.blur_p:  # blur
            if isinstance(self.blur_sigma, (int, float)):
                sigma = self.blur_sigma
            elif isinstance(self.blur_sigma, tuple) and \
                    len(self.blur_sigma) == 2:
                sigma = random.uniform(*self.blur_sigma)
            elif isinstance(self.blur_sigma, list):
                sigma = random.choice(self.blur_sigma)
            else:
                raise ValueError("sigma not understood")
            if sigma > 1e-3:  # skip computations if sigma is negligible
                for channel in sm.xrange(c):
                    image[:, :, :, channel] = ndimage.gaussian_filter(
                        image[:, :, :, channel], sigma, mode="mirror")

        if self.fliplr and self.fliplr_p <= random.random():
            image = image[:, ::-1, ...].copy()

        return image

    def set_crop(self, mode, size):
        if mode not in ['uniform', 'center', None]:
            raise ValueError("mode must be one of 'uniform', 'center' or None")
        self.crop_mode = mode
        self.crop_size = size

    def set_blur(self, sigma, p=0.5):
        self.blur_sigma = sigma
        self.blur_p = p

    def set_fliplr(self, fliplr: bool, p=0.5):
        self.fliplr = fliplr
        self.fliplr_p = p

    def _compute_crop_sample_space(self, z, y, x):
        space = range(self.crop_size//2, z - self.crop_size//2 - 1), \
                range(self.crop_size//2, y - self.crop_size//2 - 1), \
                range(self.crop_size//2, x - self.crop_size//2 - 1)
        return space

    def __str__(self):
        string = "An augmentor that will perform: "
        operations = []
        if self.crop_mode is not None and self.crop_size is not None:
            operations.append("cropping")
        if self.blur_sigma is not None and self.blur_p != 0:
            operations.append("blurring")
        if self.fliplr and self.fliplr_p != 0:
            operations.append("left-right flipping")
        string += ', '.join(operations)
        return string






















































