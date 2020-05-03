"""
This module contains the implementation of reading from different sources
to the Pandas DataFrame.

Datatype to operate on:
    Pandas DataFrame.

Todo:
    Accepting info in online mode.

Contact info:
Antonina Bondarchuk (c)
antonina.bondarchuk@nure.ua
2020
"""

from pandas import read_csv


FIRST_POSSIBLE_COLUMN_VAL = '0'


def read_file_to_df(data_path, delimiter=',', header=None, columns_to_drop=None):
    """
    Implements simplified and generalized reading from csv file.
    Ignores defined columns.
    Args:
        data_path (str): absolute path to the data source.
        delimiter (str): symbol to separate values while reading.
        header (bool/None): if 1st line of file contains columns' headers.
        columns_to_drop (str): sequence of columns numbers
            separated by comma and space to ignore. Starts from 0.
            Example:
                >>> '0, 13, 6'

    References:
        pandas.read_csv

    Returns:
        Pandas DataFrame.
    """
    dataframe = read_csv(filepath_or_buffer=data_path,
                         delimiter=delimiter,
                         header=header)
    last_column_num = str(list(dataframe)[-1])
    if columns_to_drop:
        columns_nums_to_drop = [int(col)
                                for col in columns_to_drop.split(', ')
                                if FIRST_POSSIBLE_COLUMN_VAL <= col <= last_column_num]
        dataframe = dataframe.drop(columns_nums_to_drop, axis=1)
    return dataframe
