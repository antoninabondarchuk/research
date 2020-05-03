"""
This module contains the implementation of comparing two DataFrames.

Main function uses implementation of calculating:
    1) Delta values between input DataFrame and
       Dataframe after compression and decompression.
    2) Maximum delta values by column.
    3) General maximum value in the whole DataFrame.
    4) Percentage of loss by maximum delta value subtraction.

Datatype to operate on:
    Pandas DataFrame.

Contact info:
Antonina Bondarchuk (c)
antonina.bondarchuk@nure.ua
2020
"""


def get_delta(dataframe1, dataframe2):
    """
    Calculates absolute delta values between the dataframes.
    Args:
        dataframe1 (Pandas DataFrame): DataFrame before applying operations.
        dataframe2 (Pandas DataFrame): DataFrame after applying operations.

    Returns:
        Pandas DataFrame with absolute delta values.

    Raises:
        TypeError: if input DataFrame is empty.
    """
    if dataframe1.empty or dataframe2.empty:
        raise TypeError('Cannot calculate the delta values '
                        'on None objects.')
    return abs(dataframe1 - dataframe2)


def get_max_columns_delta(delta_df):
    """
    Calculates maximum delta values per column in the dataframe.
    Args:
        delta_df (Pandas DataFrame): delta values.

    Returns:
        Pandas DataFrame with size [columns_num x 1].

    Raises:
        TypeError: if input DataFrame is empty.
    """
    if delta_df.empty:
        raise TypeError('It is impossible to calculate '
                        'maximum delta values on the None object.')
    return delta_df.max()


def get_max_delta(column_delta_df):
    """
    Calculates absolute maximum delta value in the whole dataframe.
    Args:
        column_delta_df (Pandas DataFrame): maximum deltas by columns.

    Returns:
        Float.

    Raises:
        TypeError: if input DataFrame is empty.
    """
    if column_delta_df.empty:
        raise TypeError('It is impossible to calculate '
                        'maximum delta value on the None object.')
    return max(column_delta_df)


def get_percentage(max_loss=0.):
    """
    Calculates percentage delta of decompression accuracy.
    Args:
        max_loss (float): maximum value of loss through the DataFrame.

    Returns:
        Float decompression accuracy value.

    """
    return 100 - max_loss


def get_statistics(df1, df2):
    """
    Calculates set of statistics metrics to measure quality of
    compression by calculating:
        1) Delta values between input DataFrame and
           Dataframe after operations
           (e.g. compression and decompression).
        2) Maximum delta values by column.
        3) General maximum value in the whole DataFrame.
        4) Percentage of loss by maximum delta value subtraction.
    Args:
        df1: Pandas DataFrame before operating.
        df2: Pandas DataFrame after operations
        (e.g. compression and decompression).
    References:
        get_delta, get_max_columns_delta, get_max_delta, get_percentage.
        They can raise TypeError in case of empty input DataFrames.

    Returns:
        Tuple (Pandas DataFrame, Pandas Dataframe, float, float)
        responding to the mentioned calculations.

    Raises:
        TypeError: if one of the input DataFrames is empty.
    """
    if df1.empty or df2.empty:
        raise TypeError('To get statistics for df1 and df2 delta, '
                        'please, check if they are not None.')

    delta_df = get_delta(df1, df2)
    columns_deltas_df = get_max_columns_delta(delta_df)

    # results of loss
    max_delta = get_max_delta(columns_deltas_df)
    lowest_accuracy = get_percentage(max_delta)

    return delta_df, columns_deltas_df, max_delta, lowest_accuracy
