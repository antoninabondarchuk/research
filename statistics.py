def get_delta(df1, df2):
    return abs(df1 - df2)


def get_max_columns_delta(delta_df):
    return delta_df.max()


def get_max_delta(column_delta_df):
    return max(column_delta_df)


def get_percentage(df):
    return 100 - df


def get_statistics(df1, df2):
    delta_df = get_delta(df1, df2)
    columns_deltas_df = get_max_columns_delta(delta_df)

    # results of loss
    max_delta = get_max_delta(columns_deltas_df)
    lowest_accuracy = get_percentage(max_delta)

    return delta_df, columns_deltas_df, max_delta, lowest_accuracy

