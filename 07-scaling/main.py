import pickle
import pandas as pd
import seaborn as sns
from PIL.ImageOps import scale
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt

# Custom for python script

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

with open('../06-transformation/rfmd_transformed.pkl', 'rb') as file:
    RFMD_transformed = pickle.load(file)

# END

# standardize RFMD features
scaler = RobustScaler()
RFMD_scaled_robust = scaler.fit_transform(RFMD_transformed[['recency', 'frequency', 'monetary']])

scaler = StandardScaler()
RFMD_scaled_standard = scaler.fit_transform(RFMD_transformed[['recency', 'frequency', 'monetary']])

# plot the results of both scaling method (robust, standard)
robust = pd.DataFrame(RFMD_scaled_robust, columns=['recency', 'frequency', 'monetary'])
standard = pd.DataFrame(RFMD_scaled_standard, columns=['recency', 'frequency', 'monetary'])

fig, axes = plt.subplots(2, 3, figsize=(20, 15))

sns.set_context("notebook", font_scale=2)

# before scaling
sns.boxplot(x='recency', data=RFMD_transformed, ax=axes[0, 0])
axes[0, 0].set_title('Recency Boxplot (Before Scaling)')

sns.boxplot(x='frequency', data=RFMD_transformed, ax=axes[0, 1])
axes[0, 1].set_title('Frequency Boxplot (Before Scaling)')

sns.boxplot(x='monetary', data=RFMD_transformed, ax=axes[0, 2])
axes[0, 2].set_title('Monetary Boxplot (Before Scaling)')

# standard scaler
sns.boxplot(x='recency', data=standard, ax=axes[1, 0])
axes[1, 0].set_title('Recency Boxplot (StandardScaler)')

sns.boxplot(x='frequency', data=standard, ax=axes[1, 1])
axes[1, 1].set_title('Frequency Boxplot (StandardScaler)')

sns.boxplot(x='monetary', data=standard, ax=axes[1, 2])
axes[1, 2].set_title('Monetary Boxplot (StandardScaler)')

# increase font size of the axes labels
for ax in axes.flat:
    ax.set_xlabel(ax.get_xlabel(), fontsize=20)
    ax.set_ylabel(ax.get_ylabel(), fontsize=20)

# increase font size of the tick labels
for ax in axes.flat:
    ax.tick_params(axis='both', labelsize=15)

plt.tight_layout()
# plt.show()
plt.savefig('scaling_comparison.png', dpi=300, bbox_inches='tight')

# choose which scaler to use based on it's performance
RFMD_scaled = pd.DataFrame(RFMD_scaled_standard, columns=['recency', 'frequency', 'monetary'])

print(RFMD_scaled.head())
print(RFMD_scaled.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T)

# Custom for python script

RFMD_scaled.to_pickle('rfmd_scaled.pkl')

# END