"""This module contains the base class for feature extractors."""
import abc


class BaseExtractor:
    """Base class for feature extractors."""

    def __init__(self):
        pass

    def fit(self, x):
        """fit"""

    @abc.abstractmethod
    def transform(self, x):
        """transform"""

    def fit_transform(self, x):
        """fit transform"""
