"""
This module contains the implementation of some Data Preprocessing step.
Read more here:
    https://towardsdatascience.com/data-preprocessing-concepts-fa946d11c825
See Also:
    https://en.wikipedia.org/wiki/Data_pre-processing

Main function uses implementation of:
    1) Filling the Null values.
    2) Coding values in the interval (hypercube: [-1; 1]).
    3) Centering values.

Datatype to operate on:
    Pandas DataFrame.

Contact info:
Antonina Bondarchuk (c)
antonina.bondarchuk@nure.ua
2020
"""

from numpy import mean, nan, float as float_


def fill_na_vals(dataframe, null_values):
    """
    Fills null values in DataFrame.
    Args:
        dataframe (Pandas DataFrame): raw data.
        null_values (str): sequence of symbols to mark null values in data.
            Example: >>> '?, Nan, NA, N/a, NaN'
            Note: '' need to be checked.

    Returns:
        Pandas DataFrame without Null values.

    Raises:
        TypeError: if the input DataFrame is empty.
    """
    if dataframe.empty:
        raise TypeError('It is impossible to fill Null values'
                        'in the empty dataframe.')
    null_vals_list = null_values.split(', ')
    nan_df = dataframe
    for val in null_vals_list:
        nan_df.replace(to_replace=val, value=nan, inplace=True)
    nan_df = nan_df.astype(float_)
    result_df = nan_df.fillna(nan_df.mean())
    return result_df


def hypercube(dataframe):
    """
    Transforms data by coding it on interval [-1, 1] inclusively, where -1 and 1 correspond
    to the minimum and maximum values in columns.
    Read more info: https://en.wikipedia.org/wiki/Unit_interval
    Args:
        dataframe (Pandas DataFrame): data without Null values.

    Returns:
        Pandas DataFrame with all values in [-1, 1].

    Raises:
        TypeError: if the input DataFrame is empty.
    """
    if dataframe.empty:
        raise TypeError('It is impossible to uniform on hypercube'
                        'the empty dataframe.')
    columns_min = dataframe.min()
    columns_max = dataframe.max()
    result_df = (2 * (dataframe - columns_min) / (columns_max - columns_min)) - 1
    return result_df


def center(dataframe):
    """
    Centering earlier transformed data by dividing the arithmetic mean values
    by column.
    Args:
        dataframe (Pandas DataFrame): transfromed data to center.

    Returns:
        Pandas DataFrame.

    Raises:
        TypeError: if the input DataFrame is empty.
    """
    if dataframe.empty:
        raise TypeError('It is impossible to center data'
                        'of the empty dataframe.')
    columns_mean = mean(dataframe)
    result_df = dataframe - columns_mean
    return result_df


def prepare(dataframe, null_values=None):
    """
    Preprocessing data to further operations and applying algorithms.
    Args:
        dataframe (Pandas DataFrame): raw data.
        null_values (str): sequence of symbols to mark null values in data.
            Example: >>> '?, Nan, NA, N/a, NaN'
            Note: '' need to be checked.

    Returns:
        Pandas DataFrame.

    Raises:
        TypeError: if the input DataFrame is empty.
    """
    if dataframe.empty:
        raise TypeError('It is impossible to preprocess data'
                        'of the empty dataframe.')
    if null_values is not None:
        dataframe = fill_na_vals(dataframe, null_values)
    hypercubed_df = hypercube(dataframe)
    centered_df = center(hypercubed_df)
    return centered_df
