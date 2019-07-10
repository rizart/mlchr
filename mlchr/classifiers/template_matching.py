"""This module contains template matching classification algorithms."""
import copy
import sys
from scipy.spatial import distance


class TemplateMatchingClassifier:
    """TemplateMatchingClassifier"""

    def __init__(self):
        """
        Sets distance functions that can be used.
        """
        self.distance_map = {
            'jaccard': distance.jaccard,
            'yule': distance.yule
        }
        self.X = None
        self.y = None

    def fit(self, x, y):
        """
        :param x: A list of 2d numpy arrays that represent pixel values(0/1).
        :param y: A list of the corresponding target classes for the images.
        """

        self.y = y
        # flatten 2d arrays
        self.X = copy.deepcopy(x)
        self.X = [img.flatten() for img in self.X]

    def predict(self, x_test, dist='yule', print_progress=False):
        """
        :param x_test: A list of 2d numpy arrays that holds images.
        :param dist: Distance ('yule' or 'jaccard').
        :param print_progress: Print prediction progress (False by default)
        :return: A list with the predicted target classes.
        """

        # set distance function
        distance_function = self.distance_map[dist]

        # flatten 2d arrays
        x_t = copy.deepcopy(x_test)
        x_t = [img.flatten() for img in x_t]

        y = []
        i = 0
        for img in x_t:
            dissimilarity = sys.maxsize
            class_prediction = None

            if print_progress:
                sys.stdout.write("\rTemplate Matching [{1}]:{0}%".format(
                    int(((i + 1) / len(x_t)) * 100), dist))
                sys.stdout.flush()
                i += 1

            for j in range(0, len(self.X)):
                temp_dissimilarity = distance_function(img, self.X[j])

                if temp_dissimilarity < dissimilarity:
                    dissimilarity = temp_dissimilarity
                    class_prediction = self.y[j]

            y.append(class_prediction)

        # return predictions
        return y
