import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Custom for python script

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

with open('../01-preprocessing/customers_dataset.pkl', 'rb') as file:
    customers_dataset = pickle.load(file)

with open('../03-rfm/rfm_raw.pkl', 'rb') as file:
    RFM = pickle.load(file)

# END

customer_state_dict = customers_dataset.set_index('customer_unique_id')['customer_state'].to_dict()

RFM['State'] = RFM['customer_unique_id'].map(customer_state_dict)

RFMD_heatmap = RFM.copy()
RFMD_Raw = RFM.copy()
print(RFMD_heatmap.head())

# dropping unnecessary columns
RFMD_heatmap.drop(['segments', 'recency', 'frequency', 'M_Score', 'RFM_Score'], axis=1, inplace=True)

# re-arrange columns
RFMD_heatmap = RFMD_heatmap[['customer_unique_id', 'R_Score', 'F_Score', 'monetary', 'State']]

# rename columns
RFMD_heatmap.columns = ['Customer Unique ID', 'Recency', 'Frequency', 'Monetary', 'State']

print(RFMD_heatmap.head())

print(RFMD_heatmap.describe())

print(RFMD_heatmap.info())

# heat map
rfmd_pivot = RFMD_heatmap.pivot_table(index='Recency', columns='Frequency', values='Monetary', aggfunc='mean').sort_index(ascending=False)

plt.figure(figsize=(8, 6))
sns.heatmap(rfmd_pivot, cmap="Blues", annot=True, fmt='.0f', linewidths=.5)
plt.title("RFMD Heat Map")
plt.xlabel("Frequency")
plt.ylabel("Recency")
# plt.show()
plt.savefig("rfmd_heatmap.png", dpi=300, bbox_inches='tight')

# Create a dictionary to map state abbreviations to cardinal directions
state_direction = {
    'AC': 'North',
    'AP': 'North',
    'AM': 'North',
    'PA': 'North',
    'RO': 'North',
    'TO': 'North',
    'RR': 'North',
    'AL': 'Northeast',
    'BA': 'Northeast',
    'CE': 'Northeast',
    'MA': 'Northeast',
    'PE': 'Northeast',
    'PI': 'Northeast',
    'SE': 'Northeast',
    'PB': 'Northeast',
    'RN': 'Northeast',
    'DF': 'Mid-West',
    'GO': 'Mid-West',
    'MT': 'Mid-West',
    'MS': 'Mid-West',
    'MG': 'Southeast',
    'ES': 'Southeast',
    'RJ': 'Southeast',
    'SP': 'Southeast',
    'PR': 'South',
    'RS': 'South',
    'SC': 'South'
}

# Map the states to their corresponding cardinal directions
RFMD_heatmap['State'] = RFMD_heatmap['State'].map(state_direction)
RFMD_Raw['State'] = RFMD_Raw['State'].map(state_direction)

print(RFMD_heatmap.head())

# drop unnecessary columns
RFMD_Raw.drop(['R_Score', 'M_Score', 'F_Score', 'RFM_Score', 'segments'], axis=1, inplace=True)

print(RFMD_Raw.head())

# describe %1 %5 %10 %25 %50 %75 %90 %95 %99
print(RFMD_Raw.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T)

RFMD_Raw.to_pickle('rfmd_raw.pkl')