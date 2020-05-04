"""
This module contains the implementation of the Oja's rule algorithm.
Read more here: https://en.wikipedia.org/wiki/Oja%27s_rule
Main function uses implementation of:
    1) Oja's Compression rule.
        1.1) Generating start vector w0.
        1.2) Calculating main components (y and w vectors for each).
        1.3) Dividing main components from the original data
             one by one.
    2) Oja's Decompression rule.

Datatype to operate on:
    Pandas DataFrame.

Contact info:
Antonina Bondarchuk (c)
antonina.bondarchuk@nure.ua
2020
"""

from random import uniform
import numpy as np
from pandas import DataFrame


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
    norm_start_w0 = start_w0 / np.linalg.norm(start_w0)
    return norm_start_w0


def calculate_y(dataframe_row, vector_w):
    """
    Calculates component Y value in the current iteration.
    Multiplies dataframe_row with transposed vector_w to get
    scalar value y.
    Args:
        dataframe_row (Numpy Array): data vector to multiply
        vector_w (Numpy Array): eigen vector to multiply

    Returns:
        Float.
    """
    y_val = np.dot(dataframe_row, np.transpose(vector_w))
    return y_val


def calculate_w(dataframe_row, prev_w, prev_y, df_len):
    """
    Calculates vector w for current iteration.
    Using formula from Oja's rule.
    Read more here: https://en.wikipedia.org/wiki/Oja%27s_rule
    Args:
        dataframe_row (Numpy Array): data row to compress.
        prev_w (Numpy Array): vector W on the previous iteration.
        prev_y (float): Y value on the previous iteration.
        df_len (int): quantity of rows in the compressing DataFrame.

    Returns:
        Vector W as Numpy Array.
    """
    vector_w = prev_w + (prev_y / df_len * (dataframe_row - prev_y * prev_w))
    norm_vector_w = vector_w / np.linalg.norm(vector_w)
    return norm_vector_w.to_numpy()


def calculate_component(dataframe, vector_w, component_num):
    """
    Calculates vector component Y and eigen vector W.
    Args:
        dataframe (Pandas DataFrame): preprocessed data.
        vector_w (Numpy Array): start eigen vector w0.
        component_num (int): power of 10 to calculate iterations num,
            number of component to calculate.

    Returns:
        Tuple:
        (Last eigen vector W as List,
         component values Y as List).

    Raises:
        TypeError: if the input DataFrame is empty.
    """
    if dataframe.empty:
        raise TypeError('It is impossible to calculate eigen vector W '
                        'and component Y on the empty dataframe.')

    df_size = len(dataframe)
    # calculate start value y(1)
    y_val = calculate_y(dataframe.iloc[0], vector_w)

    # to reach the stable state of the component
    # it should be calculated 10^component_num times.
    for _ in range(10 ** component_num):
        y_vector = [y_val, ]
        for row in range(1, df_size):
            vector_w = calculate_w(dataframe.iloc[row], vector_w,
                                   y_vector[row - 1], df_size)
            y_val = calculate_y(dataframe.iloc[row], vector_w)
            y_vector.append(y_val)

    component = (y_vector, vector_w)
    return component


def subtract_component(dataframe, component_y, vector_w):
    """
    Subtracts main component dataframe from the original one.
    Args:
        dataframe (Pandas DataFrame):
        component_y (Numpy Array): vector [dataframe_len x 1] of components y.
        vector_w (Numpy Array): eigen vector w [1 x df_columns_len].

    Returns:
        Pandas DataFrame.

    Raises:
        TypeError: if the input DataFrame is empty.
    """
    if dataframe.empty:
        raise TypeError('It is impossible to calculate eigen vector W '
                        'and component Y on the empty dataframe.')
    component_df = np.outer(component_y, vector_w)
    result_df = dataframe - component_df
    return result_df


def compress(dataframe):
    """
    Compress data in n_components using Oja's rule.
    Read more here: https://en.wikipedia.org/wiki/Oja%27s_rule
    Args:
        dataframe (Pandas DataFrame): data to compress.

    Returns:
        Tuple:
        (Matrix of components Y [df_rows x df_columns] as Numpy Array,
         Matrix of components W [df_columns x df_columns] as Numpy Array).

    Raises:
        TypeError: if the input DataFrame is empty.
    """
    if dataframe.empty:
        raise TypeError('It is impossible to compress'
                        'the empty dataframe.')
    n_components = len(dataframe.columns)
    # generating start vector w0
    vector_w = generate_start_w0(n_components)
    y_matrix = []
    w_matrix = []
    for component_num in range(n_components):
        y_val, vector_w = calculate_component(dataframe, vector_w, component_num)
        y_matrix.append(y_val)
        w_matrix.append(vector_w)
        dataframe = subtract_component(dataframe, y_val, vector_w)
    return y_matrix, w_matrix


def decompress(matrix_y, matrix_w):
    """
    Applies Oja's decopmressing rule to the component.
    Args:
        column_y (Numpy Array): vector [dataframe_len x 1] of components y.
        vector_w (Numpy Array): eigen vectors w [1 x df_columns_len].

    Returns:
        Pandas DataFrame.
    """
    rows, cols = len(matrix_y[0]), len(matrix_y)
    result_array = np.zeros((rows, cols))
    for i in range(rows):
        for k in range(cols):
            result_array[i] = result_array[i] + matrix_w[k] * matrix_y[k][i]
    return DataFrame(result_array)


def apply_oja(dataframe):
    """
    Implements algorithm of the Oja's rule for compression
    and decompression data.
    Args:
        dataframe (Pandas DataFrame): contains data after
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
    # compression
    matrix_y, matrix_w = compress(dataframe)
    # decompression
    decompressed_df = decompress(matrix_y, matrix_w)

    return decompressed_df
