import numpy as np
import pickle
import pandas as pd

# Custom for python script

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

with open('../04-rfmd/rfmd_raw.pkl', 'rb') as file:
    RFMD_Raw = pickle.load(file)

# END

def iqr_outliers(df):
    # Create a copy of the DataFrame to avoid modifying the original
    df_clean = df.copy()

    # Numeric columns to analyze
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Outlier detection results
    outlier_results = {}

    # Descriptive statistics with multiple percentiles
    desc_stats = df.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T

    # Outlier detection for each numeric column
    for column in numeric_columns:
        # Calculate IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate bounds
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        # Find outliers
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]

        # Store results
        outlier_results[column] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Outliers Count': len(outliers),
            'Outlier Percentage': (len(outliers) / len(df)) * 100
        }

        # Remove outliers from the clean DataFrame
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]

    return desc_stats, outlier_results, df_clean

def z_score_outliers(df):
    # Create a copy of the DataFrame to avoid modifying the original
    df_clean = df.copy()

    # Numeric columns to analyze
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Outlier detection results
    outlier_results = {}

    # Outlier detection for each numeric column
    for column in numeric_columns:
        # Calculate Z-scores
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())

        # Find outliers using DataFrame's boolean indexing
        outlier_mask = z_scores > 3
        outliers = df.loc[outlier_mask, column]

        # Store results
        outlier_results[column] = {
            'Outliers Count': len(outliers),
            'Outlier Percentage': (len(outliers) / len(df)) * 100
        }

        # Remove outliers from the clean DataFrame
        df_clean = df_clean.loc[~outlier_mask]

    # Descriptive statistics with multiple percentiles
    desc_stats = df_clean.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T

    return desc_stats, outlier_results, df_clean

def print_outlier_analysis(df, outlier_results, df_clean):
    print("--- Outlier Analysis ---")
    for column, results in outlier_results.items():
        print(f"Column: {column}")
        print(f"Outliers Count: {results['Outliers Count']}")
        print(f"Outlier Percentage: {results['Outlier Percentage']:.2f}%\n")


    print("--- Descriptive Statistics ---")
    print(f"Original DataFrame size: {len(df)}")
    print(f"Cleaned DataFrame size: {len(df_clean)}")
    print(f"Total Rows Removed: {len(df) - len(df_clean)}")

# outlier detection using Z-Score
desc_stats, outlier_results, df_clean = z_score_outliers(RFMD_Raw)

# describe
print(desc_stats)

# outlier analysis
print_outlier_analysis(RFMD_Raw, outlier_results, df_clean)

# print frequency and it's count
print(df_clean['frequency'].value_counts().sort_index(ascending=False))

print(df_clean.head())

print(df_clean.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T)

df_clean.to_pickle('rfmd_clean.pkl')