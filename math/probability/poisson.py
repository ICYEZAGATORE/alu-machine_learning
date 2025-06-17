#!/usr/bin/env python3
"""Defines the Poisson distribution class"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes a Poisson distribution.

        Args:
            data (list): List of data to estimate the distribution.
            lambtha (float): Expected number of occurrences in a time frame.

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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the PMF for a given number of 'successes'.

        Args:
            k (int): Number of successes

        Returns:
            float: PMF value for k
        """
        k = int(k)
        if k < 0:
            return 0

        num = (self.lambtha ** k) * self._exp(-self.lambtha)
        denom = self._factorial(k)
        return num / denom

    def cdf(self, k):
        """
        Calculates the CDF for a given number of 'successes'.

        Args:
            k (int): Number of successes

        Returns:
            float: CDF value for k
        """
        k = int(k)
        if k < 0:
            return 0

        e_term = self._exp(-self.lambtha)
        cdf_sum = 0
        factorial = 1
        power = 1

        for i in range(k + 1):
            if i > 0:
                factorial *= i
                power *= self.lambtha
            else:
                factorial = 1
                power = 1
            cdf_sum += (power * e_term) / factorial

        return cdf_sum

    def _factorial(self, n):
        """
        Computes factorial of a number.

        Args:
            n (int): The number

        Returns:
            int: n!
        """
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

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
