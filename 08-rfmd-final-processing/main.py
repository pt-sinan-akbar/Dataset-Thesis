import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

with open('../05-outlier/rfmd_clean.pkl', 'rb') as file:
    df_clean = pickle.load(file)
with open('../07-scaling/rfmd_scaled.pkl', 'rb') as file:
    RFMD_scaled = pickle.load(file)

# Label Encoding

# do label encoding for state
state_encoder = LabelEncoder()
# state_encoded = state_encoder.fit_transform(RFMD_Raw['State'])
state_encoded = state_encoder.fit_transform(df_clean['State'])


state_df = pd.DataFrame(state_encoded, columns=['State'])

# print each state and their corresponding encoded value
state_mapping = dict(zip(state_encoder.classes_, state_encoder.transform(state_encoder.classes_)))

# make a dataframe of the mapping
state_mapping_df = pd.DataFrame(list(state_mapping.items()), columns=['State', 'Code'])

print(state_mapping_df)

# Map the encoded state to the original state

# decode the state
encoded_to_state = {v: k for k, v in state_mapping.items()}

# plot state with legend of each state
plt.figure(figsize=(10, 6))
sns.countplot(x='State', data=state_df, palette='viridis', hue=state_df['State'].map(encoded_to_state))
plt.title('State Distribution')
plt.xlabel('State')
plt.ylabel('Count')
plt.tight_layout()
# plt.show()
plt.savefig('state_distribution.png', dpi=300, bbox_inches='tight')

# merge all features
RFMD_final = pd.concat([RFMD_scaled, state_df], axis=1)
RFMD_final['Code'] = RFMD_final['State'].map(encoded_to_state)
print(RFMD_final.head())

# describe RFMD for 1% 5% 10% 25% 50% 75% 90% 95% 99%
print(RFMD_final.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T)

# check if RFMD_final has any missing values
print(RFMD_final.isnull().sum())

# the missing values are from the state column, because the state and it's code doesn't get affected when the writer do outlier removal
# RFMD_final.dropna(inplace=True)

# check if RFMD_final has any missing values
# RFMD_final.isnull().sum()

# splitting into numerical and categorical
RFM_numerical = pd.DataFrame(RFMD_final, columns=['recency', 'frequency', 'monetary'])
RFM_categorical = RFMD_final['State']

pickle.dump(encoded_to_state, open('encoded_to_state.pkl', 'wb'))
pickle.dump(state_mapping, open('state_mapping.pkl', 'wb'))
RFMD_final.to_pickle('rfmd_final.pkl')
RFM_numerical.to_pickle('rfm_numerical.pkl')
RFM_categorical.to_pickle('rfm_categorical.pkl')