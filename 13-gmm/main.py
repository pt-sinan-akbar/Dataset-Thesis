import pickle
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from utils import plot_3d_clusters, summarize_cluster

# Custom for python script

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

with open('../08-rfmd-final-processing/rfmd_final.pkl', 'rb') as file:
    RFMD_final = pickle.load(file)
with open('../08-rfmd-final-processing/rfm_numerical.pkl', 'rb') as file:
    RFM_numerical = pickle.load(file)
with open('../05-outlier/rfmd_clean.pkl', 'rb') as file:
    df_clean = pickle.load(file)

# END

### Find the optimal number of clusters using the BIC and AIC criteria

gaussian_df = RFM_numerical.copy()

n_components = range(1, 11)
model = [GaussianMixture(n_components=i, random_state=42).fit(gaussian_df) for i in n_components]

# AICS (Akaike Information Criterion)
aics = [m.aic(gaussian_df) for m in model]

# BICS (Bayesian Information Criterion)
bics = [m.bic(gaussian_df) for m in model]

### Plot AIC and BIC

plt.plot(n_components, aics, label='AIC', marker='o')
plt.plot(n_components, bics, label='BIC', marker='o')
plt.title('AIC and BIC for Gaussian Mixture Models')
plt.xlabel('Number of Components')
plt.ylabel('Information Criterion')
plt.xticks(n_components)
plt.legend()
plt.grid()
# plt.show()
plt.savefig('aic_bic_gmm.png', dpi=300, bbox_inches='tight')

gmm = GaussianMixture(n_components=2, random_state=42)
labels = gmm.fit_predict(gaussian_df)

# Add the cluster labels to the data
data_with_clusters = gaussian_df.copy()
data_with_clusters['Cluster'] = labels

# raw
gmm_rfm_raw = pd.DataFrame(df_clean, columns=["recency", "frequency", "monetary", "State"])
gmm_rfm_raw['Cluster'] = labels
print(gmm_rfm_raw.head())

# Create a 3D scatter plot
plot_3d_clusters(data_with_clusters, "Gaussian Mixture Model")

# add the cluster labels to the data
gmm_data_with_categorical = pd.concat([data_with_clusters, RFMD_final['State']], axis=1)

# add the cluster labels to the data
gmm_data_with_categorical['Cluster'] = labels
data_with_clusters['Cluster'] = labels

# display cluster unique values on cluster with categorical
print("GMM cluster summary:")
summarize_cluster(gmm_data_with_categorical)

# cluster summary with raw data
print("GMM cluster summary (Original Data):")
summarize_cluster(gmm_rfm_raw, False)