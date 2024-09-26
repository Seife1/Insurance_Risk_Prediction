import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.use('Agg')


def univariate_analysis(data, num_cols=None, cat_cols=None):
    figs = []
    if num_cols is None:
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if cat_cols is None:
        cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Histograms for Numerical Columns
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.histplot(data[col].dropna(), kde=True, bins=30, color='royalblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution of {col}', fontsize=18, fontweight='bold', color='navy')
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend([col], loc='upper right', fontsize=12)
        plt.tight_layout()
        figs.append(fig)

    # Bar Charts for Categorical Columns
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = sns.color_palette("coolwarm", len(data[col].unique()))
        sns.countplot(x=col, data=data, hue=col, legend=False, palette=colors, order=data[col].value_counts().index)
        ax.set_title(f'Distribution of {col}', fontsize=18, fontweight='bold', color='darkred')
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        figs.append(fig)

    return figs


def scatter_plot(data, x_col, y_col, hue_col=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col)
    ax.set_title(f'Scatter Plot of {x_col} vs {y_col}')
    plt.tight_layout()
    return fig

def correlation_matrix(data, cols):
    
    corr_matrix = data[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    return fig
    

def plot_geographical_trends(data, cover_types):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Filter the data to include only these cover types
    filtered_data = data[data['CoverType'].isin(cover_types)]

    # 1. Cover Type Distribution by Province (bar plot)
    sns.countplot(x='Province', hue='CoverType', data=filtered_data, palette='Set3', ax=axs[0, 0])
    axs[0, 0].set_title('Distribution of Common Cover Types Across Provinces')
    axs[0, 0].set_xlabel('Province')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].tick_params(axis='x', rotation=45)
    axs[0, 0].legend(title='Cover Type', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    # 2. Car Make Distribution by Province (bar plot)
    car_make_counts = data.groupby('Province')['make'].count().reset_index()
    sns.barplot(x='Province', y='make', data=car_make_counts, ax=axs[0, 1])
    axs[0, 1].set_title('Car Make Distribution by Province')
    axs[0, 1].set_xlabel('Province')
    axs[0, 1].set_ylabel('Count of Car Makes')
    axs[0, 1].tick_params(axis='x', rotation=45)

    # 3. Total Premium by Province (box plot)
    sns.boxplot(x='Province', y='TotalPremium', data=data, showmeans=True, ax=axs[1, 0])
    axs[1, 0].set_title('Distribution of Total Premium by Province')
    axs[1, 0].set_xlabel('Province')
    axs[1, 0].set_ylabel('Total Premium')
    axs[1, 0].tick_params(axis='x', rotation=45)

    # 4. Vehicle Type Distribution by Province (count plot)
    sns.countplot(x='Province', hue='VehicleType', data=data, palette='Set1', ax=axs[1, 1])
    axs[1, 1].set_title('Vehicle Type Distribution by Province')
    axs[1, 1].set_xlabel('Province')
    axs[1, 1].set_ylabel('Count of Vehicle Types')
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].legend(title='Vehicle Type', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    return fig


def plot_outliers_boxplot(data, cols):
    # If a single column is passed as a string, convert it to a list
    if isinstance(cols, str):
        cols = [cols]
    
    # Create subplots
    fig, ax = plt.subplots(1, len(cols), figsize=(12, 4))
    
    # If only one column, 'ax' is not a list, so we need to handle it differently
    if len(cols) == 1:
        ax = [ax]  # Convert single AxesSubplot to a list for consistent indexing
    
    # Plot the boxplots
    for i, col in enumerate(cols):
        sns.boxplot(y=data[col], color='lightblue', ax=ax[i])
        ax[i].set_title(f'Box Plot of {col}')
    
    plt.tight_layout()
    return fig


def cap_all_outliers(data, numerical_columns):
    
    for column in numerical_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap the outliers
        data[column] = data[column].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    
    return data


def plot_violin_premium_by_cover(data, x_col, y_col):
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.violinplot(x=x_col, y=y_col, data=data, inner='quartile')
    ax.set_title('Distribution of TotalPremium by CoverType')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
    
def plot_pairplot(data, cols):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.pairplot(data[cols], palette='coolwarm')
    ax.set_title('Pair Plot of Key Numerical Features')
    plt.tight_layout()
    return fig
    
def plot_correlation_heatmap(data, cols):
    
    fig, ax = plt.subplots(figsize=(8, 4))
    corr_matrix = data[cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', linewidths=0.5)
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    return fig

def cover_type_vis(data):
    cover_type_counts = data['CoverType'].value_counts()

    # Create a bar chart with a color palette
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(x=cover_type_counts.index, y=cover_type_counts, 
                hue=cover_type_counts.index, legend=False,palette='viridis')
    ax.set_title('Cover Type Frequencies')
    ax.set_xlabel('Cover Type')
    ax.set_ylabel('Count')
    plt.xticks(rotation=90)  # Rotate labels to the bottom
    return fig
