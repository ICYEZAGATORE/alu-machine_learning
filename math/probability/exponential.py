#!/usr/bin/env python3
"""Defines the Exponential distribution class"""


class Exponential:
    """Represents an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes an exponential distribution.

        Args:
            data (list): List of data to estimate the distribution.
            lambtha (float): Expected number of occurrences per time unit.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data has fewer than two values.
            ValueError: If lambtha is not a positive value.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            self.lambtha = 1 / mean

    def pdf(self, x):
        """
        Calculates the PDF for a given time period.

        Args:
            x (float): Time period

        Returns:
            float: PDF value for x
        """
        if x < 0:
            return 0
        return self.lambtha * self._exp(-self.lambtha * x)

    def _exp(self, x):
        """
        Approximates e^x using a Taylor series expansion.

        Args:
            x (float): The exponent

        Returns:
            float: Approximate value of e^x
        """
        result = 1.0
        term = 1.0
        for i in range(1, 100):
            term *= x / i
            result += term
            if abs(term) < 1e-10:
                break
        return result
