import pickle
import pandas as pd
from sklearn.cluster import KMeans
from utils import plot_3d_clusters, summarize_cluster

with open('../08-rfmd-final-processing/rfmd_final.pkl', 'rb') as file:
    RFMD_final = pickle.load(file)
with open('../08-rfmd-final-processing/rfm_numerical.pkl', 'rb') as file:
    RFM_numerical = pickle.load(file)
with open('../05-outlier/rfmd_clean.pkl', 'rb') as file:
    df_clean = pickle.load(file)

# K-Means menggunakan RFM
# berdasarkan hasil elbow diatas, jumlah kluster yang akan dipilih adalah 4
kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(RFM_numerical)

data_with_clusters = RFM_numerical.copy()

# add the cluster labels to the data
knn_data_with_categorical = pd.concat([data_with_clusters, RFMD_final['State']], axis=1)

# add the cluster labels to the data
knn_data_with_categorical['Cluster'] = kmeans.labels_
data_with_clusters['Cluster'] = kmeans.labels_

plot_3d_clusters(data_with_clusters, "K-Means")

# raw
print("raw row count:", len(df_clean))
print("kemans row count:", len(kmeans.labels_))
kmeans_rfm_raw = pd.DataFrame(df_clean, columns=["recency", "frequency", "monetary", "State"])
kmeans_rfm_raw['Cluster'] = kmeans.labels_
print(kmeans_rfm_raw.head())

# making sure the cluster is the same as the one in knn_data_with_categorical
test = knn_data_with_categorical.groupby('State').size().reset_index(name='count')
print(test.head())
test2 = kmeans_rfm_raw.groupby('State').size().reset_index(name='count')
print(test2.head())

# display cluster unique values on cluster with categorical
print("KNN Cluster summary:")
print(summarize_cluster(knn_data_with_categorical))

print("KNN Cluster summary (Original Data):")
print(summarize_cluster(kmeans_rfm_raw, False))