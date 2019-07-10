"""
This module implements the subdivisions extractor algorithm.
For more details visit: http://users.iit.demokritos.gr/~bgat/PRHandRec2010.pdf
"""

from mlchr.feature_extraction.base import BaseExtractor
import numpy as np
import sys


class SubdivisionsExtractor(BaseExtractor):
    """SubdivisionsExtractor"""

    def __init__(self, granularity):
        """
        :param granularity: Divisions granularity.
        """
        super().__init__()
        self.granularity = granularity

    @staticmethod
    def find_index(v):
        """
        :param v: A python list.
        :return: Min index.
        """
        # get vector size
        vsize = len(v)

        # prefix sum
        prefix_sum = [None for _ in range(0, vsize)]
        prefix_sum[0] = v[0]
        for i in range(1, vsize):
            prefix_sum[i] = prefix_sum[i - 1] + v[i]

        # suffix sum
        suffix_sum = [None for _ in range(0, vsize)]
        suffix_sum[vsize - 1] = v[vsize - 1]
        for i in range(vsize - 2, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + v[i]

        # find min index
        min_index = -1
        minv = sys.maxsize
        for i in range(1, vsize - 1):
            x = abs(prefix_sum[i] - suffix_sum[i])
            if x < minv:
                minv = x
                min_index = i

        return min_index

    def find_vertical_point(self, image):
        """
        :param image: 2d numpy array.
        :return: Vertical point of division.
        """
        height = image.shape[0]
        width = image.shape[1]

        v0 = [0 for _ in range(0, width)]
        for y in range(0, width):
            for x in range(0, height):
                if image[x, y] == 1:
                    v0[y] += 1  # vertical pixels

        v1 = [
            0 if (x % 2 == 0) else v0[int(x / 2)] for x in range(0, width * 2)
        ]

        # find index that minimizes sum difference
        xq = self.find_index(v1)
        return xq + 1

    def find_horizontal_point(self, image):
        """
        :param image: 2d numpy array.
        :return: Horizontal point of division.
        """

        height = image.shape[0]
        width = image.shape[1]

        v0 = [0 for _ in range(0, height)]
        for x in range(0, height):
            for y in range(0, width):
                if image[x, y] == 1:
                    v0[x] += 1  # horizontal pixels

        v1 = [
            0 if (x % 2 == 0) else v0[int(x / 2)]
            for x in range(0, height * 2)
        ]

        # find index that minimizes sum difference
        yq = self.find_index(v1)
        return yq + 1

    def rec_sub_div(self, image, granularity, img_features):
        """
        :param img_features: Image features so far.
        :param image: 2d numpy array.
        :param granularity: Divisions granularity.
        """

        height = image.shape[0]
        width = image.shape[1]

        # image can't be divided any further,
        # just fill remaining features with (0,0)
        if height < 3 or width < 3:
            for _ in range(0, 4**granularity):
                img_features.append(0)
                img_features.append(0)
            return

        xq = self.find_vertical_point(image)
        x0 = int(xq / 2)

        yq = self.find_horizontal_point(image)
        y0 = int(yq / 2)

        # left up sub-image
        rows = [x for x in range(0, y0)]
        columns = [x for x in range(0, x0)]
        left_up_image = image[np.ix_(rows, columns)]

        # right up sub-image
        rows = [x for x in range(0, y0)]
        if xq % 2 == 0:
            columns = [x for x in range(x0 - 1, width)]
        else:
            columns = [x for x in range(x0, width)]
        right_up_image = image[np.ix_(rows, columns)]

        # left down sub-image
        if yq % 2 == 0:
            rows = [x for x in range(y0 - 1, height)]
        else:
            rows = [x for x in range(y0, height)]
        columns = [x for x in range(0, x0)]
        left_down_image = image[np.ix_(rows, columns)]

        # right down sub-image
        if yq % 2 == 0:
            rows = [x for x in range(y0 - 1, height)]
        else:
            rows = [x for x in range(y0, height)]
        if xq % 2 == 0:
            columns = [x for x in range(x0 - 1, width)]
        else:
            columns = [x for x in range(x0, width)]
        right_down_image = image[np.ix_(rows, columns)]

        if granularity > 0:
            self.rec_sub_div(left_up_image, granularity - 1, img_features)
            self.rec_sub_div(right_up_image, granularity - 1, img_features)
            self.rec_sub_div(left_down_image, granularity - 1, img_features)
            self.rec_sub_div(right_down_image, granularity - 1, img_features)
        else:
            img_features.append(x0)
            img_features.append(y0)

    def transform(self, x):
        """
        :param x: A list of 2d numpy arrays that represent pixel values(0/1).
        :return: A numpy array of subdivision points for each image.
        """

        x_transformed = []

        # for each image
        for image in x:
            img_features = []
            self.rec_sub_div(image, self.granularity, img_features)
            x_transformed.append(img_features)

        return np.array(x_transformed)
