#!/usr/bin/env python3
"""Poisson Distribution class - no imports allowed"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Class constructor"""
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
        """Calculates the PMF for a given number of 'successes'"""
        k = int(k)
        if k < 0:
            return 0

        return (self.lambtha ** k * self._exp(-self.lambtha)) / self._factorial(k)

    def _factorial(self, n):
        """Calculates factorial of n (n!)"""
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def _exp(self, x):
        """Approximates e^x using a Taylor series"""
        result = 1.0
        term = 1.0
        for i in range(1, 50):  # 50 terms for decent precision
            term *= x / i
            result += term
        return result
    
