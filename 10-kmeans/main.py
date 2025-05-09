import pandas as pd
from sklearn.cluster import KMeans

import utils
from utils import plot_3d_clusters, summarize_cluster
from logger import Logger
from benchmark import Benchmark

# Custom for python script

utils.widen_output(pd)
RFMD_final = utils.import_pickle('../08-rfmd-final-processing/rfmd_final.pkl')
RFM_numerical = utils.import_pickle('../08-rfmd-final-processing/rfm_numerical.pkl')
df_clean = utils.import_pickle('../05-outlier/rfmd_clean.pkl')
logger = Logger()
benchmark = Benchmark(logger)

# END

logger.print("K-Means using RFM")
benchmark.start_benchmark()
# berdasarkan hasil elbow diatas, jumlah kluster yang akan dipilih adalah 4
kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(RFM_numerical)
benchmark.end_benchmark()

data_with_clusters = RFM_numerical.copy()

# add the cluster labels to the data
knn_data_with_categorical = pd.concat([data_with_clusters, RFMD_final['State']], axis=1)

# add the cluster labels to the data
knn_data_with_categorical['Cluster'] = kmeans.labels_
data_with_clusters['Cluster'] = kmeans.labels_

plot_3d_clusters(data_with_clusters, "K-Means")

# raw
logger.print("raw row count:", len(df_clean))
logger.print("kemans row count:", len(kmeans.labels_))
kmeans_rfm_raw = pd.DataFrame(df_clean, columns=["recency", "frequency", "monetary", "State"])
kmeans_rfm_raw['Cluster'] = kmeans.labels_
logger.print("Head of kmeans_rfm_raw:")
logger.print(kmeans_rfm_raw.head())

logger.print("making sure the cluster is the same as the one in knn_data_with_categorical")
test = knn_data_with_categorical.groupby('State').size().reset_index(name='count')
logger.print(test.head())
test2 = kmeans_rfm_raw.groupby('State').size().reset_index(name='count')
logger.print(test2.head())

# display cluster unique values on cluster with categorical
logger.print("KNN Cluster summary:")
logger.print(summarize_cluster(knn_data_with_categorical))

logger.print("KNN Cluster summary (Original Data):")
logger.print(summarize_cluster(kmeans_rfm_raw, False))

utils.summarize_cluster_v2(kmeans_rfm_raw)

logger.print("evaluation metrics")
eval_results = utils.evaluation_metrics(
    df=knn_data_with_categorical,
    algorithm="KMeans",
    cluster_range=range(2, 7),
)
logger.print("Evaluation results:")
logger.print(eval_results)

logger.print("Plot evaluation metrics")
utils.plot_evaluation_metrics(eval_results)


utils.export_pickle(knn_data_with_categorical, "rfm_kmeans.pkl")
utils.export_pickle(kmeans_rfm_raw, "rfm_kmeans_clean.pkl")
utils.export_pickle(eval_results, "rfm_kmeans_eval.pkl")
