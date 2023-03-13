import random
import numpy as np


class Augmentor():
    def __init__(self):
        self.crop_mode = None
        self.crop_size = None

    def augment(self, image: np.ndarray):
        crop_slice = slice(None)  # 'crop' to the whole image

        if self.crop_mode is not None and self.crop_size is not None:  # crop
            z, y, x, c = image.shape  # assuming z*y*x*c shape
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

        # TODO: more augmentations
        return image[crop_slice]

    def set_crop(self, mode, size):
        if mode not in ['uniform', 'center', None]:
            raise ValueError("mode must be one of 'uniform', 'center' or None")
        self.crop_mode = mode
        self.crop_size = size

    def _compute_crop_sample_space(self, z, y, x):
        # TODO: take other augmentations into account
        pass
        space = range(z//2 - self.crop_size//2, z//2 + self.crop_size//2), \
            range(y//2 - self.crop_size//2, y//2 + self.crop_size//2), \
            range(x//2 - self.crop_size//2, x//2 + self.crop_size//2)
        return space

    def __str__(self):
        string = "An augmentor that will perform: "
        operations = []
        if self.crop_mode is not None and self.crop_size is not None:
            operations.append("cropping")
        string += ', '.join(operations)
        return string






















































