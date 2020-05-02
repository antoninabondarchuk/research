from numpy import mean, dot
from pandas import DataFrame
from sklearn.decomposition import PCA


def apply_pca(df, n_components=2):
    mu = mean(df, axis=0)

    pca = PCA(n_components=n_components)
    pca.fit(df)

    # recompression
    result_arr = dot(pca.transform(df)[:, :n_components], pca.components_[:n_components, :])
    result_arr += mu
    result_df = DataFrame(result_arr, columns=list(df))
    return result_df
