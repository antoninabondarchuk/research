from numpy import mean, nan, float as float_


def fill_na_vals(df, null_values):
    null_vals_list = null_values.split(' ')
    nan_df = df
    for val in null_vals_list:
        nan_df.replace(to_replace=val, value=nan, inplace=True)
    nan_df = nan_df.astype(float_)
    result_df = nan_df.fillna(nan_df.mean())
    return result_df


def hypercube(df):
    columns_min = df.min()
    columns_max = df.max()
    result_df = (2 * (df - columns_min) / (columns_max - columns_min)) - 1
    return result_df


def center(df):
    columns_mean = mean(df)
    result_df = df - columns_mean
    return result_df


def prepare(df, null_values=None):
    if null_values is not None:
        df = fill_na_vals(df, null_values)
    hypercubed_df = hypercube(df)
    centered_df = center(hypercubed_df)
    return centered_df
