"""This module implements the projections extractor algorithm."""
from mlchr.feature_extraction.base import BaseExtractor
import numpy as np


class ProjectionsExtractor(BaseExtractor):
    """ProjectionsExtractor"""

    def __init__(self, projections):
        """
        :param projections: Number of projections.
        """
        super().__init__()
        self.projections = projections

    def transform(self, x):
        """
        :param x: A list of 2d numpy arrays that represent pixel values(0/1).
        :return: A numpy array of projections for each image.
        """

        x_transformed = []

        # for each image
        for image in x:
            img_features = []
            # for each projection
            for k in range(1, self.projections + 1):
                horizontal_pixels = 0
                vertical_pixels = 0
                height = image.shape[0]
                width = image.shape[1]
                for i in range(0, int(k * height / self.projections)):
                    for y in range(0, width):
                        horizontal_pixels += image[i, y]
                        vertical_pixels += image[y, i]
                img_features.append(horizontal_pixels)
                img_features.append(vertical_pixels)

            x_transformed.append(img_features)

        return np.array(x_transformed)
