import pandas as pd
from sklearn.cluster import KMeans

import utils
from utils import plot_3d_clusters, summarize_cluster

# Custom for python script

utils.widen_output(pd)
RFMD_final = utils.import_pickle('../08-rfmd-final-processing/rfmd_final.pkl')
RFM_numerical = utils.import_pickle('../08-rfmd-final-processing/rfm_numerical.pkl')
df_clean = utils.import_pickle('../05-outlier/rfmd_clean.pkl')

# END

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
print("Head of kmeans_rfm_raw:")
print(kmeans_rfm_raw.head())

print("making sure the cluster is the same as the one in knn_data_with_categorical")
test = knn_data_with_categorical.groupby('State').size().reset_index(name='count')
print(test.head())
test2 = kmeans_rfm_raw.groupby('State').size().reset_index(name='count')
print(test2.head())

# display cluster unique values on cluster with categorical
print("KNN Cluster summary:")
print(summarize_cluster(knn_data_with_categorical))

print("KNN Cluster summary (Original Data):")
print(summarize_cluster(kmeans_rfm_raw, False))

utils.summarize_cluster_v2(kmeans_rfm_raw)

utils.export_pickle(knn_data_with_categorical, "rfm_kmeans.pkl")
utils.export_pickle(kmeans_rfm_raw, "rfm_kmeans_clean.pkl")
