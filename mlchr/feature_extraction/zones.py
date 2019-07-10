"""This module implements the zones extractor algorithm."""
from mlchr.feature_extraction.base import BaseExtractor
import numpy as np


class ZonesExtractor(BaseExtractor):
    """ZonesExtractor"""

    def __init__(self, zones, is_weighted=False, weight=1.25):
        """
        :param zones: Shape of zones (zones x zones).
        :param is_weighted: Indicate to use weighted zones.
        :param weight: Indicate weight (used only if is_weighted=True, must be >1).
        """
        super().__init__()
        self.zones = zones
        self.is_weighted = is_weighted
        self.weight = weight

    def transform(self, x):
        """
        :param x: A list of 2d numpy arrays that represent pixel values(0/1).
        :return: A numpy array of zones for each image.
        """

        x_transformed = []

        # for each image
        for image in x:
            img_features = []
            height = image.shape[0]
            width = image.shape[1]
            # offsets external loops
            for i in range(0, int(height / self.zones)):
                for j in range(0, int(width / self.zones)):
                    # calculate pixel density in zone
                    pixels = 0
                    for xi in range(i * self.zones, (i + 1) * self.zones):
                        for yj in range(j * self.zones, (j + 1) * self.zones):
                            pixels += image[xi, yj]
                            if self.is_weighted:
                                if xi < int(height / self.zones / 4):
                                    if image[xi, yj] == 1:
                                        pixels += self.weight
                    img_features.append(pixels)

            x_transformed.append(img_features)

        return np.array(x_transformed)


class AdaptiveZonesExtractor(BaseExtractor):
    """
    http://users.iit.demokritos.gr/~bgat/ICDAR2011_AdaptZoning.pdf
    """

    def __init__(self, zones, adj_range=2):
        """
        :param zones: Shape of zones (zones x zones).
        :param adj_range: Adjustment range.
        """
        super().__init__()
        self.zones = zones
        self.adj_range = adj_range

    def transform(self, x):
        """
        :param x: A list of 2d numpy arrays that represent pixel values(0/1).
        :return: A numpy array of zones for each image.
        """

        x_transformed = []

        # for each image
        for image in x:
            img_features = []
            height = image.shape[0]
            width = image.shape[1]
            # offsets external loops
            for i in range(0, int(height / self.zones)):
                for y in range(0, int(width / self.zones)):
                    # calculate pixel density in zone
                    zones_pixels = []
                    for xo in range(-self.adj_range, self.adj_range + 1):
                        for yo in range(-self.adj_range, self.adj_range + 1):
                            pixels = 0
                            from_xi = (i * self.zones) + xo
                            to_xi = ((i + 1) * self.zones) + xo
                            from_yi = (y * self.zones) + yo
                            to_yi = ((y + 1) * self.zones) + yo
                            if from_xi < 0 or to_xi > height - 1:
                                from_xi -= xo
                                to_xi -= xo
                            if from_yi < 0 or to_yi > width - 1:
                                from_yi -= yo
                                to_yi -= yo
                            for xi in range(from_xi, to_xi):
                                for yi in range(from_yi, to_yi):
                                    pixels += image[xi, yi]
                            zones_pixels.append(pixels)

                    img_features.append(max(zones_pixels))

            x_transformed.append(img_features)

        return np.array(x_transformed)
