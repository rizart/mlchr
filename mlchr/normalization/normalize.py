"""This module contains the normalizer class."""
from PIL import Image


class Normalizer:
    """Normalizer"""

    def __init__(self, size):
        """
        :param size: Output size of normalized images.
        """
        self.size = size

    def pillow_resize(self, images, img_filter=Image.NEAREST):
        """
        :param images: List of OCRImage images.
        :param img_filter: Resampling filter to apply during normalization.
        For more available filters see:
        https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters
        """
        # normalize images
        for img in images:
            img.pil_image = img.pil_image.resize((self.size, self.size),
                                                 img_filter)
            img.create_matrix()
