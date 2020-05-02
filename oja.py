from random import uniform
from numpy import linalg


def generate_start_w0(columns_num):
    start_w0 = [uniform(-1, 1.) for _ in range(columns_num)]
    norm_start_w0 = start_w0 / linalg.norm(start_w0)
    return norm_start_w0


def apply_oja(df):
    start_w0 = generate_start_w0(len(df.columns))

    return
