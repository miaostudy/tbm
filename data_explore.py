import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy import stats
import os
import logging
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_datetime64_dtype


# TODO: Automatic analysis of statistical information
def generate_statistics_feature(data):
    data.info()
    logging.info(data.describe())


def generate_data_description(df):
    sample_count = df.shape[0]
    feature_count = df.shape[1]
    first_feature = df.columns[0]
    last_feature = df.columns[-1]
    feature_range = f"{first_feature}-{last_feature}"

    numeric_count = 0
    string_count = 0
    datetime_count = 0
    other_count = 0

    for col in df.columns:
        if is_numeric_dtype(df[col]):
            numeric_count += 1
        elif is_string_dtype(df[col]) or df[col].dtype == 'object':
            string_count += 1
        elif is_datetime64_dtype(df[col]):
            datetime_count += 1
        else:
            other_count += 1

    type_parts = []
    if numeric_count > 0:
        type_parts.append(f"{numeric_count} numeric")
    if string_count > 0:
        type_parts.append(f"{string_count} string/object")
    if datetime_count > 0:
        type_parts.append(f"{datetime_count} datetime")
    if other_count > 0:
        type_parts.append(f"{other_count} other type")

    type_description = ", ".join(type_parts)

    missing_count = df.isnull().sum().sum()
    missing_description = f"no missing values" if missing_count == 0 else f"{missing_count} missing values"

    description = (f"This training dataset contains a total of {sample_count} samples, with {feature_count} "
                   f"feature variables ranging from {feature_range}. The variables consist of {type_description} types. "
                   f"In total, the data features have {missing_description}.")

    return description

def main(args):
    # read data
    if args.path.endswith('.csv'):
        data = pd.read_csv(args.path)
    elif args.path.endswith('.xlsx'):
        data = pd.read_excel(args.path)
    else:
        raise ValueError('Please provide a csv or xlsx data path')

    generate_statistics_feature(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='csv or xlsx data path')
    parser.add_argument('--log_dir', type=str, default='./logs', help='log path')

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(args.log_dir, 'data_explore_log.txt'), filemode='w')

    main(args)
