"""This module contains a customized Pillow Image class."""
import numpy as np


class OCRImage:
    """OCRImage"""

    def __init__(self,
                 pil_image,
                 img_id=None,
                 img_class=None,
                 matrix=None,
                 img_hex=None):
        """
        :param pil_image: Pillow Image.
        https://pillow.readthedocs.io/en/stable/reference/Image.html
        :param img_id: Unique id for image object.
        :param img_class: Image class.
        :param matrix: 2d numpy array with image pixels (0/1).
        """
        self.img_id = img_id
        self.img_class = img_class
        self.img_hex = img_hex
        self.matrix = matrix
        self.pil_image = pil_image.convert('1')
        self.width, self.height = self.pil_image.size

    def create_matrix(self):
        """
        Create 2d array of binary image
        """
        self.matrix = np.array(self.pil_image)
        self.matrix = (1 - self.matrix).astype(int)

    def print(self):
        """
        Print image matrix in ascii
        """
        for row in range(0, self.height):
            for column in range(0, self.width):
                print(self.matrix[row, column], end=' ')
            print("")

    def show(self):
        """
        Show image
        """
        self.pil_image.show()
