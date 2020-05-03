"""
This module contains the implementation of the Oja's rule algorithm.
Read more here: https://en.wikipedia.org/wiki/Oja%27s_rule
Main function uses implementation of:
    1) Oja's rule Compression.
        1.1) Generating start vector w0.
        1.2)
    2) Oja's rule Decompression.

Datatype to operate on:
    Pandas DataFrame.

Contact info:
Antonina Bondarchuk (c)
antonina.bondarchuk@nure.ua
2020
"""

from random import uniform
from numpy import linalg


def generate_start_w0(columns_num):
    """
    Generating start vector with random numbers in range [-1; 1]
    with length columns_num. Using division by norm of this vector
    to scale values.
    Args:
        columns_num (int): length of the vector w0.

    Returns:
        Numpy Array.
    """
    start_w0 = [uniform(-1, 1.) for _ in range(columns_num)]
    norm_start_w0 = start_w0 / linalg.norm(start_w0)
    return norm_start_w0


def apply_oja(dataframe):
    """
    Implements algorithm of the Oja's rule for compression
    and decompression data.
    Args:
        dataframe: Pandas DataFrame containing data after
            data preprocessing stage to compress.
    Raises:
        TypeError: in case applying function on the empty DataFrame.

    Returns:
        Decompressed Pandas DataFrame.

    Raises:
        TypeError: if the input DataFrame is empty.
    """
    if dataframe.empty:
        raise TypeError("It is impossible to apply the Oja's rule"
                        "on the empty dataframe.")
    start_w0 = generate_start_w0(len(dataframe.columns))

    return
