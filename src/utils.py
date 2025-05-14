from typing import List, Optional
import pandas as pd
import numpy as np

# Constants
NUMERIC_DTYPE = 'number'
CATEGORICAL_DTYPES = ['object', 'category', 'bool']
BIN_PREFIX = 'bin_'
CATEGORICAL_PREFIX_SEPARATOR = '='


def create_bin_labels(num_bins: int) -> List[str]:
    """
    Generate a list of bin labels based on the specified number of bins.

    This function creates and returns a list of bin labels formatted as strings.
    Each label is prefixed with the constant `BIN_PREFIX` followed by the
    corresponding bin number, starting from 1 up to the number of bins
    specified by the `num_bins` parameter.

    :param num_bins: The number of bin labels to generate.
    :type num_bins: int
    :return: A list of bin labels formatted as strings.
    :rtype: List[str]
    """
    return [f"{BIN_PREFIX}{i + 1}" for i in range(num_bins)]


def booleanize_numeric_columns(df: pd.DataFrame, num_bins: int = 3) -> pd.DataFrame:
    """
    Transforms numeric columns in a DataFrame into boolean indicator columns based on their
    binned quantile ranges. Each numeric column is divided into `num_bins` bins representing
    quantile ranges, and for each bin, a new column is created with boolean values indicating
    whether the respective value belongs to that bin.

    :param df: The input DataFrame.
    :type df: pd.DataFrame
    :param num_bins: The number of bins to divide the numeric columns into. Defaults to 3.
    :type num_bins: int, optional
    :return: A new DataFrame with boolean indicator columns for each numeric column and bin.
    :rtype: pd.DataFrame
    """
    result = pd.DataFrame(index=df.index)
    numeric_columns = df.select_dtypes(include=NUMERIC_DTYPE).columns

    for column in numeric_columns:
        quantiles = [df[column].quantile(i / num_bins) for i in range(1, num_bins)]
        unique_quantiles = sorted(set(quantiles))
        bin_labels = create_bin_labels(len(unique_quantiles) + 1)

        bin_edges = [-np.inf] + unique_quantiles + [np.inf]
        binned_data = pd.cut(df[column], bins=bin_edges, labels=bin_labels)

        for bin_label in bin_labels:
            result[f"{column}_{bin_label}"] = (binned_data == bin_label).astype(int)

    return result


def process_dataset_for_fca(df: pd.DataFrame,
                            num_bins: int = 3,
                            drop_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Processes a dataset for Formal Concept Analysis (FCA) by preparing it in a
    booleanized format. The dataset is processed by separating numeric and
    categorical columns, then booleanizing each column type appropriately.
    Numeric columns are transformed using bins, and categorical columns are
    converted to boolean representations. Columns specified in the excluded
    list will be dropped from the dataset.

    :param df: The source dataset to process.
    :type df: pd.DataFrame

    :param num_bins: The number of bins to use for booleanization of numeric
        columns, default is 3.
    :type num_bins: int, optional

    :param drop_columns: A list of columns to drop from the dataset, default is
        None (no columns will be dropped).
    :type drop_columns: Optional[List[str]]

    :return: A DataFrame where all columns are transformed to boolean types
        (binary representation) suitable for FCA.
    :rtype: pd.DataFrame
    """
    if drop_columns:
        df = df.drop(columns=drop_columns, errors='ignore')

    numeric_df = df.select_dtypes(include=NUMERIC_DTYPE)
    categorical_df = df.select_dtypes(include=CATEGORICAL_DTYPES)

    boolean_numeric = booleanize_numeric_columns(numeric_df, num_bins=num_bins)
    boolean_categorical = pd.get_dummies(categorical_df,
                                         prefix_sep=CATEGORICAL_PREFIX_SEPARATOR,
                                         dtype=bool)

    return pd.concat([boolean_numeric.astype(bool),
                      boolean_categorical], axis=1)
