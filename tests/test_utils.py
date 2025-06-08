import pandas as pd
import pytest
import numpy as np
from src.utils import process_dataset_for_fca, get_support, get_confidence, get_lift


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5],
        'categorical_col': ['A', 'B', 'A', 'A', 'C'],
        'to_drop': ['X', 'Y', 'Z', 'W', 'V']
    })


@pytest.fixture
def binary_dataframe():
    return pd.DataFrame({
        'A': [1, 0, 1, 1, 0],
        'B': [0, 1, 1, 1, 0],
        'C': [0, 0, 0, 1, 1]
    })


def test_get_confidence_with_empty_base_and_add(binary_dataframe):
    base = set()
    add = set()
    result = get_confidence(binary_dataframe, base, add)
    assert result == 0.0


def test_get_confidence_with_non_empty_base_and_empty_add(binary_dataframe):
    base = {'A'}
    add = set()
    result = get_confidence(binary_dataframe, base, add)
    assert result == 0.0


def test_get_confidence_with_empty_base_and_non_empty_add(binary_dataframe):
    base = set()
    add = {'B'}
    result = get_confidence(binary_dataframe, base, add)
    assert result == 0.0


def test_get_confidence_with_valid_base_and_add(binary_dataframe):
    base = {'B'}
    add = {'A'}
    result = get_confidence(binary_dataframe, base, add)
    expected = 2 / 3
    assert np.isclose(result, expected)


def test_get_confidence_base_with_zero_support(binary_dataframe):
    base = {'C'}
    add = {'A'}
    result = get_confidence(binary_dataframe, base, add)
    assert result == 0.5


def test_get_confidence_with_combined_support(binary_dataframe):
    base = {'A'}
    add = {'B'}
    result = get_confidence(binary_dataframe, base, add)
    expected = 2 / 3
    assert np.isclose(result, expected)


def test_process_dataset_with_default_bins_and_no_drop(sample_dataframe):
    processed_df = process_dataset_for_fca(sample_dataframe)
    assert not processed_df.empty
    assert any(col.startswith('numeric_col_bin_') for col in processed_df.columns)
    assert 'categorical_col=A' in processed_df.columns
    assert 'categorical_col=B' in processed_df.columns
    assert 'categorical_col=C' in processed_df.columns


def test_process_dataset_with_drop_columns(sample_dataframe):
    processed_df = process_dataset_for_fca(sample_dataframe, drop_columns=['to_drop'])
    assert 'to_drop' not in processed_df.columns


def test_process_dataset_with_custom_num_bins(sample_dataframe):
    processed_df = process_dataset_for_fca(sample_dataframe, num_bins=2)
    numeric_columns = [col for col in processed_df.columns if col.startswith('numeric_col')]
    assert len(numeric_columns) > 0
    assert all('_1' in col.lower() or '_2' in col.lower() for col in numeric_columns)


def test_process_dataset_with_empty_dataframe():
    empty_df = pd.DataFrame()
    processed_df = process_dataset_for_fca(empty_df)
    assert processed_df.empty


def test_get_support_with_valid_itemset(binary_dataframe):
    itemset = {'A', 'B'}
    result = get_support(binary_dataframe, itemset)
    assert result == 2 / 5


def test_get_support_with_empty_itemset(binary_dataframe):
    itemset = set()
    result = get_support(binary_dataframe, itemset)
    assert result == 0.0


def test_get_support_with_single_column(binary_dataframe):
    itemset = {'C'}
    result = get_support(binary_dataframe, itemset)
    assert result == 2 / 5


def test_get_lift_with_independent_features(binary_dataframe):
    base = {'A'}
    add = {'C'}
    result = get_lift(binary_dataframe, base, add)
    expected = 5 / 6
    assert np.isclose(result, expected)


def test_get_lift_with_same_features(binary_dataframe):
    base = {'A'}
    add = {'A'}
    result = get_lift(binary_dataframe, base, add)
    expected = 5/3
    assert np.isclose(result, expected)


def test_get_lift_with_positive_lift(binary_dataframe):
    base = {'B'}
    add = {'A'}
    result = get_lift(binary_dataframe, base, add)
    assert result > 1.0


def test_get_support_with_no_matching_rows(binary_dataframe):
    itemset = {'A', 'C'}
    result = get_support(binary_dataframe, itemset)
    assert result == 0.2
