"""This module contains an Artificial Neural Network classifier."""
import tensorflow as tf


class ANNClassifier:
    """CNNClassifier"""

    def __init__(self, num_of_classes, learning_rate=0.01):
        """
        :param num_of_classes: Number of target classes.
        :param learning_rate: Learning rate for GradientDescentOptimizer.
        """

        self.learning_rate = learning_rate
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(784, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2000, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2000, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_of_classes, activation=tf.nn.softmax)
        ])

        # set model configurations
        self.model.compile(optimizer=tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, x, y, epochs=10, batch_size=64, verbose=0):
        """
        :param x: A list of 2d numpy arrays that represent pixel values(0/1).
        :param y: A list of the corresponding target classes for the images.
        :param epochs: Training epochs.
        """

        self.model.fit(x,
                       y,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose)

    def predict(self, x_test):
        """
        :param x_test: A list of 2d numpy arrays that holds images.
        :return: A list with the predicted target classes.
        """

        preds = self.model.predict(x_test)
        y = []
        for pred in preds:
            y.append(pred.argmax(axis=0))

        # return predictions
        return y
