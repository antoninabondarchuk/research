"""
This module contains the implementation of Principal Component Analysis.
Read more here: https://en.wikipedia.org/wiki/Principal_component_analysis
Main function to use is apply_pca, which includes compression as well
as decompression step.

Datatype to operate on:
    Pandas DataFrame.

Note:
    Compression is used from sklearn.decomposition.PCA module.

Contact info:
Antonina Bondarchuk (c)
antonina.bondarchuk@nure.ua
2020
"""

from numpy import mean, dot
from pandas import DataFrame
from sklearn.decomposition import PCA


def pca_compression(dataframe, n_components=2):
    """
    Compress data using Principal Component Analysis algorithm.
    Read more here: https://en.wikipedia.org/wiki/Principal_component_analysis
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    Args:
        dataframe (Pandas DataFrame): contains data after Data Preprocessing stage.
        n_components (int): number of the components to calculate.
    Note:
        To get principal components use pca.components_.

    Returns:
        sklearn.decomposition._pca.PCA.

    Raises:
        TypeError: if input DataFrame is empty.
    """
    if dataframe.empty:
        raise TypeError('It is impossible to apply PCA compression '
                        'on the empty DataFrame.')
    pca = PCA(n_components=n_components)
    pca.fit(dataframe)
    return pca


def pca_decompression(dataframe, pca, n_components=2):
    """
    Implements decompression using PCA decompression calculations.
    Read more info:
    https://stats.stackexchange.com/questions/454814/is-decompression-possible-with-pca
    Args:
        dataframe (Pandas DataFrame): compressed with PCA.
        pca (sklearn.decomposition._pca.PCA): to reach transform and components
            of the compressed DataFrame.
        n_components (int): number of the components to get.

    Returns:
        Decompressed Pandas DataFrame.

    Raises:
        TypeError: if input DataFrame is empty.
    """
    if dataframe.empty:
        raise TypeError('It is impossible to apply PCA decompression '
                        'on the empty DataFrame.')
    df_mean = mean(dataframe, axis='rows')
    result_arr = dot(pca.transform(dataframe)[:, :n_components],
                     pca.components_[:n_components, :])
    result_arr += df_mean
    result_df = DataFrame(result_arr, columns=list(dataframe))
    return result_df


def apply_pca(dataframe, n_components=2):
    """
    Implements Principal Component Analysis compression and decompression.
    Read more here: https://en.wikipedia.org/wiki/Principal_component_analysis
    Args:
        dataframe (Pandas DataFrame): contains data after Data Preprocessing stage.
        n_components (int): number of the components to calculate.
    References:
        pca_compression, pca_decompression.

    Returns:
        Pandas DataFrame.

    Raises:
        TypeError: if input DataFrame is empty.
    """
    if dataframe.empty:
        raise TypeError('It is impossible to apply PCA compression '
                        'and decompression on the empty DataFrame.')
    # compression
    pca = pca_compression(dataframe, n_components)

    # decompression
    decompressed_df = pca_decompression(dataframe, pca, n_components)
    return decompressed_df
