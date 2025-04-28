import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt

# Custom for python script

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

with open('../05-outlier/rfmd_clean.pkl', 'rb') as file:
    df_clean = pickle.load(file)
    
# END

# check the original skewness of the data
frequency_skew = df_clean['frequency'].skew()
monetary_skew = df_clean['monetary'].skew()
recency_skew = df_clean['recency'].skew()

sns.set_theme(style='whitegrid')

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

sns.histplot(df_clean['frequency'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title(f'Frequency Distribution (Skewness: {frequency_skew:.2f})', fontsize=14)
axes[0].set_xlabel('Frequency')
axes[0].set_ylabel('Count')
axes[0].text(0.5, 0.5, f'Skewness: {frequency_skew:.2f}', transform=axes[0].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

sns.histplot(df_clean['monetary'], kde=True, ax=axes[1], color='salmon')
axes[1].set_title(f'Monetary Distribution (Skewness: {monetary_skew:.2f})', fontsize=14)
axes[1].set_xlabel('Monetary')
axes[1].set_ylabel('Count')
axes[1].text(0.5, 0.5, f'Skewness: {monetary_skew:.2f}', transform=axes[1].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

sns.histplot(df_clean['recency'], kde=True, ax=axes[2], color='lightgreen')
axes[2].set_title(f'Recency Distribution (Skewness: {recency_skew:.2f})', fontsize=14)
axes[2].set_xlabel('Recency')
axes[2].set_ylabel('Count')
axes[2].text(0.5, 0.5, f'Skewness: {recency_skew:.2f}', transform=axes[2].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

plt.tight_layout()
# plt.show()
plt.savefig('distributions_skewness.png', dpi=300, bbox_inches='tight')

RFMD_features = df_clean[['recency', 'frequency', 'monetary']].copy()
pt = PowerTransformer(method='box-cox')
pt_yj = PowerTransformer(method='yeo-johnson')

def feature_skewness(df, feature):
    # Get columns that contain the feature string
    matching_cols = [col for col in df.columns if feature in col]

    if not matching_cols:
        print(f"No columns found matching '{feature}'")
        return None

    # Calculate skewness for matching columns
    skewness_values = df[matching_cols].skew()

    # Create a DataFrame with feature names and skewness values
    skewness_df = pd.DataFrame({'Feature': matching_cols, 'Skewness': skewness_values.values})

    # Sort by skewness value
    return skewness_df.sort_values(by='Skewness', ascending=False)

# Frequency transformation

# log1p
RFMD_features['frequency_log1p'] = np.log1p(RFMD_features['frequency'])
frequency_skew_log1p = RFMD_features['frequency_log1p'].skew()

# log 10
if RFMD_features['frequency'].min() > 0:
    RFMD_features['frequency_log10'] = np.log10(RFMD_features['frequency'])
    frequency_skew_log10 = RFMD_features['frequency_log10'].skew()
else:
    RFMD_features['frequency_log10'] = 0
    frequency_skew_log10 = RFMD_features['frequency_log10'].skew()

# boxcox
if RFMD_features['frequency'].min() > 0:
    RFMD_features['frequency_boxcox'] = pt.fit_transform(RFMD_features[['frequency']].values.reshape(-1, 1)).flatten()
    frequency_skew_boxcox = RFMD_features['frequency_boxcox'].skew()
else:
    RFMD_features['frequency_boxcox'] = 0
    frequency_skew_boxcox = RFMD_features['frequency_boxcox'].skew()

# yeo-johnson
RFMD_features['frequency_yeoh'] = pt_yj.fit_transform(RFMD_features[['frequency']].values.reshape(-1, 1)).flatten()
frequency_skew_yeoh = RFMD_features['frequency_yeoh'].skew()

# square root
if RFMD_features['frequency'].min() > 0:
    RFMD_features['frequency_sqrt'] = np.sqrt(RFMD_features['frequency'])
    frequency_skew_sqrt = RFMD_features['frequency_sqrt'].skew()
else:
    RFMD_features['frequency_sqrt'] = 0
    frequency_skew_sqrt = RFMD_features['frequency_sqrt'].skew()

# cube root 
RFMD_features['frequency_cube'] = np.cbrt(RFMD_features['frequency'])
frequency_skew_cube = RFMD_features['frequency_cube'].skew()

print(feature_skewness(RFMD_features, 'frequency'))

# plot all frequency transformation
fig, axes = plt.subplots(2, 3, figsize=(16, 12))

# log1p
sns.histplot(RFMD_features['frequency_log1p'], kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Log1p Frequency Distribution', fontsize=14)
axes[0, 0].set_xlabel('Frequency')
axes[0, 0].set_ylabel('Count')
axes[0, 0].text(0.5, 0.5, f'Skewness: {frequency_skew_log1p:.2f}', transform=axes[0, 0].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

# cube root
sns.histplot(RFMD_features['frequency_cube'], kde=True, ax=axes[0, 1], color='salmon')
axes[0, 1].set_title('Cube Frequency Distribution', fontsize=14)
axes[0, 1].set_xlabel('Frequency')
axes[0, 1].set_ylabel('Count')
axes[0, 1].text(0.5, 0.5, f'Skewness: {frequency_skew_cube:.2f}', transform=axes[0, 1].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

# log10
sns.histplot(RFMD_features['frequency_log10'], kde=True, ax=axes[0, 2], color='lightgreen')
axes[0, 2].set_title('Log10 Frequency Distribution', fontsize=14)
axes[0, 2].set_xlabel('Frequency')
axes[0, 2].set_ylabel('Count')
axes[0, 2].text(0.5, 0.5, f'Skewness: {frequency_skew_log10:.2f}', transform=axes[0, 2].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

# boxcox
sns.histplot(RFMD_features['frequency_boxcox'], kde=True, ax=axes[1, 0], color='skyblue')
axes[1, 0].set_title('Box-Cox Frequency Distribution', fontsize=14)
axes[1, 0].set_xlabel('Frequency')
axes[1, 0].set_ylabel('Count')
axes[1, 0].text(0.5, 0.5, f'Skewness: {frequency_skew_boxcox:.2f}', transform=axes[1, 0].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

# yeo-johnson
sns.histplot(RFMD_features['frequency_yeoh'], kde=True, ax=axes[1, 1], color='salmon')
axes[1, 1].set_title('Yeo-Johnson Frequency Distribution', fontsize=14)
axes[1, 1].set_xlabel('Frequency')
axes[1, 1].set_ylabel('Count')
axes[1, 1].text(0.5, 0.5, f'Skewness: {frequency_skew_yeoh:.2f}', transform=axes[1, 1].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

# square root
sns.histplot(RFMD_features['frequency_sqrt'], kde=True, ax=axes[1, 2], color='lightgreen')
axes[1, 2].set_title('Square Root Frequency Distribution', fontsize=14)
axes[1, 2].set_xlabel('Frequency')
axes[1, 2].set_ylabel('Count')
axes[1, 2].text(0.5, 0.5, f'Skewness: {frequency_skew_sqrt:.2f}', transform=axes[1, 2].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

plt.tight_layout()
# plt.show()
plt.savefig('frequency_transformations.png', dpi=300, bbox_inches='tight')

# based of the skewness, the best transformation is yeo-johnson
RFMD_features['frequency_selected'] = RFMD_features['frequency_yeoh']

# Monetary transformation

# log1p
RFMD_features['monetary_log1p'] = np.log1p(RFMD_features['monetary'])
monetary_skew_log1p = RFMD_features['monetary_log1p'].skew()

# log 10
if RFMD_features['monetary'].min() > 0:
    RFMD_features['monetary_log10'] = np.log10(RFMD_features['monetary'])
    monetary_skew_log10 = RFMD_features['monetary_log10'].skew()
else:
    RFMD_features['monetary_log10'] = 0
    monetary_skew_log10 = RFMD_features['monetary_log10'].skew()

# boxcox
if RFMD_features['monetary'].min() > 0:
    RFMD_features['monetary_boxcox'] = pt.fit_transform(RFMD_features[['monetary']].values.reshape(-1, 1)).flatten()
    monetary_skew_boxcox = RFMD_features['monetary_boxcox'].skew()
else:
    RFMD_features['monetary_boxcox'] = 0
    monetary_skew_boxcox = RFMD_features['monetary_boxcox'].skew()

# yeo-johnson
RFMD_features['monetary_yeoh'] = pt_yj.fit_transform(RFMD_features[['monetary']].values.reshape(-1, 1)).flatten()
monetary_skew_yeoh = RFMD_features['monetary_yeoh'].skew()

# square root
if RFMD_features['monetary'].min() > 0:
    RFMD_features['monetary_sqrt'] = np.sqrt(RFMD_features['monetary'])
    monetary_skew_sqrt = RFMD_features['monetary_sqrt'].skew()
else:
    RFMD_features['monetary_sqrt'] = 0
    monetary_skew_sqrt = RFMD_features['monetary_sqrt'].skew()

# cube root 
RFMD_features['monetary_cube'] = np.cbrt(RFMD_features['monetary'])
monetary_skew_cube = RFMD_features['monetary_cube'].skew()

print(feature_skewness(RFMD_features, 'monetary'))

# plot all monetary transformation
fig, axes = plt.subplots(2, 3, figsize=(16, 12))

# log1p
sns.histplot(RFMD_features['monetary_log1p'], kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Log1p Monetary Distribution', fontsize=14)
axes[0, 0].set_xlabel('Monetary')
axes[0, 0].set_ylabel('Count')
axes[0, 0].text(0.5, 0.5, f'Skewness: {monetary_skew_log1p:.2f}', transform=axes[0, 0].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

# cube root
sns.histplot(RFMD_features['monetary_cube'], kde=True, ax=axes[0, 1], color='salmon')
axes[0, 1].set_title('Cube Monetary Distribution', fontsize=14)
axes[0, 1].set_xlabel('Monetary')
axes[0, 1].set_ylabel('Count')
axes[0, 1].text(0.5, 0.5, f'Skewness: {monetary_skew_cube:.2f}', transform=axes[0, 1].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

# log10
sns.histplot(RFMD_features['monetary_log10'], kde=True, ax=axes[0, 2], color='lightgreen')
axes[0, 2].set_title('Log10 Monetary Distribution', fontsize=14)
axes[0, 2].set_xlabel('Monetary')
axes[0, 2].set_ylabel('Count')
axes[0, 2].text(0.5, 0.5, f'Skewness: {monetary_skew_log10:.2f}', transform=axes[0, 2].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

# boxcox
sns.histplot(RFMD_features['monetary_boxcox'], kde=True, ax=axes[1, 0], color='skyblue')
axes[1, 0].set_title('Box-Cox Monetary Distribution', fontsize=14)
axes[1, 0].set_xlabel('Monetary')
axes[1, 0].set_ylabel('Count')
axes[1, 0].text(0.5, 0.5, f'Skewness: {monetary_skew_boxcox:.2f}', transform=axes[1, 0].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

# yeo-johnson
sns.histplot(RFMD_features['monetary_yeoh'], kde=True, ax=axes[1, 1], color='salmon')
axes[1, 1].set_title('Yeo-Johnson Monetary Distribution', fontsize=14)
axes[1, 1].set_xlabel('Monetary')
axes[1, 1].set_ylabel('Count')
axes[1, 1].text(0.5, 0.5, f'Skewness: {monetary_skew_yeoh:.2f}', transform=axes[1, 1].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

# square root
sns.histplot(RFMD_features['monetary_sqrt'], kde=True, ax=axes[1, 2], color='lightgreen')
axes[1, 2].set_title('Square Root Monetary Distribution', fontsize=14)
axes[1, 2].set_xlabel('Monetary')
axes[1, 2].set_ylabel('Count')
axes[1, 2].text(0.5, 0.5, f'Skewness: {monetary_skew_sqrt:.2f}', transform=axes[1, 2].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

plt.tight_layout()
# plt.show()
plt.savefig('monetary_transformations.png', dpi=300, bbox_inches='tight')

# based of the skewness, the best transformation is yeo-johnson
RFMD_features['monetary_selected'] = RFMD_features['monetary_yeoh']

# Post transformation

frequency_skew = RFMD_features['frequency_selected'].skew()
monetary_skew = RFMD_features['monetary_selected'].skew()
recency_skew = RFMD_features['recency'].skew()

sns.set_theme(style='whitegrid')

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

sns.histplot(RFMD_features['frequency_selected'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title(f'Frequency Distribution (Skewness: {frequency_skew:.2f})', fontsize=14)
axes[0].set_xlabel('Frequency')
axes[0].set_ylabel('Count')
axes[0].text(0.5, 0.5, f'Skewness: {frequency_skew:.2f}', transform=axes[0].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

sns.histplot(RFMD_features['monetary_selected'], kde=True, ax=axes[1], color='salmon')
axes[1].set_title(f'Monetary Distribution (Skewness: {monetary_skew:.2f})', fontsize=14)
axes[1].set_xlabel('Monetary')
axes[1].set_ylabel('Count')
axes[1].text(0.5, 0.5, f'Skewness: {monetary_skew:.2f}', transform=axes[1].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

sns.histplot(RFMD_features['recency'], kde=True, ax=axes[2], color='lightgreen')
axes[2].set_title(f'Recency Distribution (Skewness: {recency_skew:.2f})', fontsize=14)
axes[2].set_xlabel('Recency')
axes[2].set_ylabel('Count')
axes[2].text(0.5, 0.5, f'Skewness: {recency_skew:.2f}', transform=axes[2].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

plt.tight_layout()
# plt.show()
plt.savefig('post_transformations_skewness.png', dpi=300, bbox_inches='tight')

# plot frequency before and after log transformation
fig, axes = plt.subplots(1, 4, figsize=(16, 6))

sns.boxplot(x='frequency', data=RFMD_features, ax=axes[0])
axes[0].set_title('Frequency Boxplot')

sns.boxplot(x='frequency_selected', data=RFMD_features, ax=axes[1])
axes[1].set_title('Yeo-Johnson Frequency Boxplot')

sns.boxplot(x='monetary', data=RFMD_features, ax=axes[2])
axes[2].set_title('Monetary Boxplot')

sns.boxplot(x='monetary_selected', data=RFMD_features, ax=axes[3])
axes[3].set_title('Yeo-Johnson Monetary Boxplot')

plt.tight_layout()
# plt.show()
plt.savefig('boxplots_comparison.png', dpi=300, bbox_inches='tight')

# describe %1 %5 %10 %25 %50 %75 %90 %95 %99
print(RFMD_features.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T)

# select the features with the least skewness
RFMD_features = RFMD_features[['recency', 'frequency_selected', 'monetary_selected']]
print(RFMD_features)

# copy df
RFMD_transformed = RFMD_features.copy()

# rename all columns
RFMD_transformed.columns = ['recency', 'frequency', 'monetary']

# describe %1 %5 %10 %25 %50 %75 %90 %95 %99
print(RFMD_transformed.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T)

# Custom for python script

RFMD_transformed.to_pickle('rfmd_transformed.pkl')

# END