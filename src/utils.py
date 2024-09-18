import pandas as pd
import numpy as np
import missingno as msno
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.stats import zscore
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')


def missing_data_summary(data):
        # Total missing values per column
        missing_data = data.isnull().sum()
        
        # Filter only columns with missing values greater than 0
        missing_data = missing_data[missing_data > 0]
        
        # Calculate the percentage of missing data
        missing_percentage = (missing_data / len(data)) * 100
        
        # Combine the counts and percentages into a DataFrame
        missing_data = pd.DataFrame({
            'Missing Count': missing_data, 
            'Percentage (%)': missing_percentage
        })
        
        # Sort by percentage of missing data
        missing_data = missing_data.sort_values(by='Percentage (%)', ascending=False)
        
        return missing_data


def visualize_missing_values(data):
    fig, ax = plt.subplots()
    msno.matrix(data)
    # plt.tight_layout()
    return fig  # Return the figure object


def replace_missing_values(data):
    # Checking missing values
    missing_values = data.isnull().sum()

    # List columns with missing values only
    missing_columns = missing_values[missing_values > 0].index.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Ensure the column names are used for indexing
    missing_categorical_columns = [col for col in missing_columns if col in categorical_features]
    missing_numerical_columns = [col for col in missing_columns if col in numerical_features]

    # Replace missing values in numerical columns
    if len(missing_numerical_columns) > 0:
        print(f"Replacing {len(missing_numerical_columns)} Numeric columns by mean value ...")
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        data[missing_numerical_columns] = imputer.fit_transform(data[missing_numerical_columns])
        print("Replacing Completed!!")
        print()

    # Replace missing values in categorical columns
    if len(missing_categorical_columns) > 0:
        print(f"Replacing {len(missing_categorical_columns)} Categorical columns by most frequent value ...")
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        data[missing_categorical_columns] = imputer.fit_transform(data[missing_categorical_columns])
        print("Replacing Completed!!")
        print()

    # Return the missing values matrix or updated DataFrame
    return data