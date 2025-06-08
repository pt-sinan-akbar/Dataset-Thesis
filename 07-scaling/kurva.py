import seaborn as sns
import matplotlib.pyplot as plt
import utils

before = utils.import_pickle('../06-transformation/rfmd_transformed.pkl')
after = utils.import_pickle('rfmd_scaled.pkl')

before_recency_skew = before['recency'].skew()
before_frequency_skew = before['frequency'].skew()
before_monetary_skew = before['monetary'].skew()
after_recency_skew = after['recency'].skew()
after_frequency_skew = after['frequency'].skew()
after_monetary_skew = after['monetary'].skew()

fig, axes = plt.subplots(2, 3, figsize=(16, 12))

sns.histplot(before['recency'], kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Recency Before', fontsize=14)
axes[0, 0].set_xlabel('Recency')
axes[0, 0].set_ylabel('Count')
axes[0, 0].text(0.5, 0.5, f'Skewness: {before_recency_skew:.2f}', transform=axes[0, 0].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

sns.histplot(after['recency'], kde=True, ax=axes[1, 0], color='salmon')
axes[1, 0].set_title('Recency After', fontsize=14)
axes[1, 0].set_xlabel('Recency')
axes[1, 0].set_ylabel('Count')
axes[1, 0].text(0.5, 0.5, f'Skewness: {after_recency_skew:.2f}', transform=axes[1, 0].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

sns.histplot(before['frequency'], kde=True, ax=axes[0, 1], color='lightgreen')
axes[0, 1].set_title('Frequency Before', fontsize=14)
axes[0, 1].set_xlabel('Frequency')
axes[0, 1].set_ylabel('Count')
axes[0, 1].text(0.5, 0.5, f'Skewness: {before_frequency_skew:.2f}', transform=axes[0, 1].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

sns.histplot(after['frequency'], kde=True, ax=axes[1, 1], color='orange')
axes[1, 1].set_title('Frequency After', fontsize=14)
axes[1, 1].set_xlabel('Frequency')
axes[1, 1].set_ylabel('Count')
axes[1, 1].text(0.5, 0.5, f'Skewness: {after_frequency_skew:.2f}', transform=axes[1, 1].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)
sns.histplot(before['monetary'], kde=True, ax=axes[0, 2], color='purple')

axes[0, 2].set_title('Monetary Before', fontsize=14)
axes[0, 2].set_xlabel('Monetary')
axes[0, 2].set_ylabel('Count')
axes[0, 2].text(0.5, 0.5, f'Skewness: {before_monetary_skew:.2f}', transform=axes[0, 2].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

sns.histplot(after['monetary'], kde=True, ax=axes[1, 2], color='brown')
axes[1, 2].set_title('Monetary After', fontsize=14)
axes[1, 2].set_xlabel('Monetary')
axes[1, 2].set_ylabel('Count')
axes[1, 2].text(0.5, 0.5, f'Skewness: {after_monetary_skew:.2f}', transform=axes[1, 2].transAxes, horizontalalignment='right', color='black', weight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('scaling_comparison_distribution_plot.png')