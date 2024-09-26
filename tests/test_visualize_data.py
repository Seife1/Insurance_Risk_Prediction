import pytest
import pandas as pd
from src.visualize_data import (
    univariate_analysis,
    scatter_plot,
    correlation_matrix,
    plot_geographical_trends,
    plot_outliers_boxplot,
    plot_violin_premium_by_cover,
    cover_type_vis
)

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Age': [23, 45, 56, 29, 30, 67],
        'Income': [30000, 45000, 50000, 25000, 60000, 70000],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'CoverType': ['TypeA', 'TypeB', 'TypeA', 'TypeB', 'TypeA', 'TypeB'],
        'Province': ['A', 'B', 'A', 'B', 'A', 'B'],
        'make': ['Toyota', 'Honda', 'Ford', 'BMW', 'Tesla', 'Chevrolet'],
        'VehicleType': ['Sedan', 'SUV', 'Truck', 'Sedan', 'SUV', 'Truck'],
        'TotalPremium': [1000, 1500, 2000, 1200, 1700, 1800]
    })

@pytest.mark.mpl_image_compare
def test_univariate_analysis(sample_data):
    figs = univariate_analysis(sample_data)
    assert figs is not None
    assert len(figs) > 0

@pytest.mark.mpl_image_compare
def test_scatter_plot(sample_data):
    fig = scatter_plot(sample_data, x_col='Age', y_col='Income')
    assert fig is not None

@pytest.mark.mpl_image_compare
def test_correlation_matrix(sample_data):
    fig = correlation_matrix(sample_data, ['Age', 'Income'])
    assert fig is not None

@pytest.mark.mpl_image_compare
def test_plot_geographical_trends(sample_data):
    cover_types = ['TypeA', 'TypeB']
    fig = plot_geographical_trends(sample_data, cover_types)
    assert fig is not None

@pytest.mark.mpl_image_compare
def test_plot_outliers_boxplot(sample_data):
    fig = plot_outliers_boxplot(sample_data, ['Income'])
    assert fig is not None

@pytest.mark.mpl_image_compare
def test_plot_violin_premium_by_cover(sample_data):
    fig = plot_violin_premium_by_cover(sample_data, x_col='CoverType', y_col='TotalPremium')
    assert fig is not None

@pytest.mark.mpl_image_compare
def test_cover_type_vis(sample_data):
    fig = cover_type_vis(sample_data)
    assert fig is not None