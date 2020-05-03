"""
This module contains the implementation of comparing Principal Component Analysis
with the Oja's rule algorithm.
Read more here: https://en.wikipedia.org/wiki/Principal_component_analysis

Main function uses implementation of:
    1) Data Preprocessing step.
    2) PCA Compression and Decompression.
    3) Oja's rule Compression and Decompression.
    4) Comparing results.

Datatype to operate on:
    Pandas DataFrame.

Note:
    PCA Compression is used from sklearn.decomposition.PCA module.

References:
    .env file, which should be placed in the root of the project and contain variables:
        DATA_FILE_PATH (required): absolute system path to the source file.
        COLUMNS_TO_DROP (optional): sequence of columns numbers (ints) to ignore
            separated by comma and space.
            Example: >>> '0, 13, 8'
        NULL_VALUES (optional): sequence of symbols to mark null values in data.
            Example: >>> '?, Nan, NA, N/a, NaN'
            Note: '' need to be checked.

Contact info:
Antonina Bondarchuk (c)
antonina.bondarchuk@nure.ua
2020
"""

import os
from statistics import get_statistics
from dotenv import load_dotenv
from oja import apply_oja
from apply_pca import apply_pca
from preprocessing import prepare
from reading import read_file_to_df


DEFAULT_NUM_COMPONENTS = '2'


if __name__ == "__main__":
    load_dotenv()

    # reading to Pandas DataFrame
    raw_input_df = read_file_to_df(
        os.getenv('DATA_FILE_PATH'),
        columns_to_drop=os.getenv('COLUMNS_TO_DROP'))

    # data preprocessing
    prepared_df = prepare(raw_input_df, null_values=os.getenv('NULL_VALUES'))

    # applying PCA
    pca_df = apply_pca(prepared_df,
                       n_components=int(os.getenv('NUM_COMPONENTS', DEFAULT_NUM_COMPONENTS)))

    # applying Oja
    oja_df = apply_oja(prepared_df)

    # getting statistics for PCA and Oja
    pca_delta_df, pca_cols_delta_df, pca_max_delta, pca_accuracy = get_statistics(prepared_df, pca_df)
    oja_delta_df, oja_cols_delta_df, oja_max_delta, oja_accuracy = get_statistics(prepared_df, oja_df)
