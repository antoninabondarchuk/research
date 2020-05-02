import os

from pandas import read_csv


def read_file_to_df(data_path, delimiter=',', header=None, columns_to_drop=None):
    df = read_csv(filepath_or_buffer=data_path,
                  delimiter=delimiter,
                  header=header)
    last_column_num = str(list(df)[-1])
    if columns_to_drop:
        columns_nums_to_drop = [int(col)
                                for col in columns_to_drop.split(' ')
                                if col <= last_column_num]
        df = df.drop(columns_nums_to_drop, axis=1)
    return df
