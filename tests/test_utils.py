import pandas as pd
import pytest
from src.utils import process_dataset_for_fca


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5],
        'categorical_col': ['A', 'B', 'A', 'A', 'C'],
        'to_drop': ['X', 'Y', 'Z', 'W', 'V']
    })


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
