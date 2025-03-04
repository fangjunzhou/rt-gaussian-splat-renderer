"""
Numpy math utils.
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function in numpy following :math:`\\frac{1}{1 + e^{-x}}`

    :param x: arbitrary ndarray.
    :return: ndarray in the same dimension after sigmoid.
    """
    return 1 / (1 + np.exp(-x))
