import pytest
import pandas as pd
import numpy as np
from src.utils import (
    missing_data_summary, 
    visualize_missing_values, 
    replace_missing_values
)

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 1, 1, 1],
        'C': ['cat', np.nan, 'dog', 'dog'],
        'D': [1, 2, 3, 4]
    })
    return data

def test_missing_data_summary(sample_data):
    missing_summary = missing_data_summary(sample_data)
    
    assert missing_summary.loc['A']['Missing Count'] == 1
    assert missing_summary.loc['A']['Percentage (%)'] == 25
    assert missing_summary.loc['B']['Missing Count'] == 1
    assert missing_summary.loc['B']['Percentage (%)'] == 25
    assert missing_summary.loc['C']['Missing Count'] == 1
    assert missing_summary.loc['C']['Percentage (%)'] == 25

def test_replace_missing_values(sample_data):
    updated_data = replace_missing_values(sample_data.copy())
    
    # Check if missing values are replaced
    assert updated_data.isnull().sum().sum() == 0

@pytest.mark.mpl_image_compare
def test_visualize_missing_values(sample_data):
    fig = visualize_missing_values(sample_data)
    assert fig is not None
