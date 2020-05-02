import os
from dotenv import load_dotenv
from oja import apply_oja
from apply_pca import apply_pca
from preprocessing import prepare
from reading import read_file_to_df
from statistics import get_statistics
from pandas import set_option

DEFAULT_NUM_COMPONENTS = 2


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
                       n_components=os.getenv('NUM_COMPONENTS', DEFAULT_NUM_COMPONENTS))

    # applying Oja
    oja_df = apply_oja(prepared_df)

    # getting statistics for PCA and Oja
    pca_delta_df, pca_columns_delta_df, pca_max_delta, pca_accuracy = get_statistics(prepared_df, pca_df)
    oja_delta_df, oja_columns_delta_df, oja_max_delta, oja_accuracy = get_statistics(prepared_df, oja_df)
