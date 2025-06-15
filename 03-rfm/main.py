import pickle
import pandas as pd
import seaborn as sns
import squarify
import matplotlib.pyplot as plt

# Custom for python script
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

with open('../01-preprocessing/merged_dataset.pkl', 'rb') as file:
    merged_ds = pickle.load(file)

# END

max_date = merged_ds['order_purchase_timestamp'].max()

RFM = merged_ds.groupby('customer_unique_id').agg({
    'order_purchase_timestamp': lambda x: (max_date - x.max()).days,
    'order_id': 'count',
    'payment_value': 'sum'
}).reset_index()

RFM.columns = ['customer_unique_id', 'recency', 'frequency', 'monetary']

print(RFM.isna().sum())

print(RFM.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T)

plt.figure(figsize=(10, 6))
sns.boxplot(x='recency', data=RFM)
plt.title('Recency Boxplot')
# plt.show()
plt.savefig('recency_boxplot.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
sns.boxplot(x='frequency', data=RFM)
plt.title('Frequency Boxplot')
# plt.show()
plt.savefig('frequency_boxplot.png', dpi=300, bbox_inches='tight')

# count each freq value desc by frequency
print(RFM['frequency'].value_counts().sort_index(ascending=False))

plt.figure(figsize=(10, 6))
sns.boxplot(x='monetary', data=RFM)
plt.title('Monetary Boxplot')
# plt.show()
plt.savefig('monetary_boxplot.png', dpi=300, bbox_inches='tight')
plt.cla()

RFM['R_Score'] = pd.qcut(RFM['recency'], 3, labels=[1, 2 ,3]).astype(str)
RFM['M_Score'] = pd.qcut(RFM['monetary'], 3, labels=[1, 2 ,3]).astype(str)
RFM['F_Score'] = RFM['frequency'].apply(lambda x: '1' if x == 1 else ('2' if x == 2 else '3'))

RFM['RFM_Score'] = RFM['R_Score'] + RFM['F_Score'] + RFM['M_Score']

def segment(x):
    if x in ['311', '312', '313']:  # High recency (oldest), low/mid frequency
        return 'Gone'
    elif x in ['111', '112', '113', '221', '231']:  # Low across board or transitioning customers
        return 'Rookies'
    elif x in ['233', '333', '323', '222', '232', '322']:  # High monetary with mid/high metrics
        return 'Valuable'
    else:  # Combining 'Regular' and 'Loyal' into one segment
        return 'Regular'

RFM['segments'] = RFM['RFM_Score'].apply(segment)
print(RFM['segments'].value_counts(normalize=True) * 100)

print(RFM.head())

print(RFM.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

segment_wise = RFM.groupby('segments').agg(RecencyMean = ('recency', 'mean'),
                                           FrequencyMean = ('frequency', 'mean'),
                                           MonetaryMean = ('monetary', 'mean'),
                                           Count = ('segments', 'count')).sort_values(by='MonetaryMean', ascending=False)

print(segment_wise)

plt.rcParams['font.family'] = 'DejaVu Sans'

fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 16)
squarify.plot(sizes=segment_wise['Count'],
              label=segment_wise.index,
              color=['gold', 'teal', 'steelblue', 'limegreen'],
              alpha=0.8,
              text_kwargs={'fontsize': 20, 'weight': 'normal', 'color': 'black'},
              )
plt.title("RFM Segments")
plt.axis('off')
plt.rc('font', size=14)
# plt.show()
plt.title("RFM Segments", fontsize=20)
plt.savefig('rfm_segments.png', dpi=300, bbox_inches='tight')

# pake score our lord sinan
rfm_crosstab = pd.crosstab(
    index=[RFM['R_Score'], RFM['F_Score']],
    columns=RFM['M_Score'],
    rownames=['Recency Score (1 = Recent)', 'Frequency Score (1 = Low)'],
    colnames=['Monetary Score (1 = Low, 3 = High)']
)

# biar bisa columnya
rfm_flat = rfm_crosstab.reset_index()

# biar enak diliat aja
rfm_flat['Row_Label'] = rfm_flat.apply(
    lambda row: f"R={row['Recency Score (1 = Recent)']} | F={row['Frequency Score (1 = Low)']}", axis=1
)

# recency (1 = recent, 3 = oldest)
# monetary(1 = low spender, 3 = high spender)
# frequency (1 = low, 3 = highest (ya gitulah))
monetary_columns = ['1', '2', '3']
table_data = rfm_flat[monetary_columns].values
row_labels = rfm_flat['Row_Label'].tolist()
column_labels = ['M=1 (Low)', 'M=2 (Mid)', 'M=3 (High)']

# Bikin tablenya gan 
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('tight')
ax.axis('off')

table = ax.table(
    cellText=table_data,
    rowLabels=row_labels,
    colLabels=column_labels,
    loc='center',
    cellLoc='center'
)

table.scale(1.2, 1.5)
table.auto_set_font_size(False)
table.set_fontsize(10)

# Add title
plt.title("Cross Tabulation", fontsize=14, pad=20)

# Show the table
# plt.show()
plt.savefig('rfm_cross_tabulation.png', dpi=300, bbox_inches='tight')

# Custom for python script
RFM.to_pickle('rfm_raw.pkl')
# END