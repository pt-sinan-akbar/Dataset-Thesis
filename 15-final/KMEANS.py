from random import random

import pandas as pd
from sklearn.cluster import KMeans
import utils
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

kmeans_df = RFM_numerical.copy()

logger.print("Running K-Means clustering")
benchmark.start_benchmark()
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(kmeans_df)
benchmark.end_benchmark()

# Add the cluster labels to the data
data_with_clusters = kmeans_df.copy()
data_with_clusters['Cluster'] = labels

# raw
kmeans_rfm_raw = pd.DataFrame(df_clean, columns=["recency", "frequency", "monetary", "State"])
kmeans_rfm_raw['Cluster'] = labels
logger.print(kmeans_rfm_raw.head())

# Create a 3D scatter plot
utils.plot_3d_clusters(data_with_clusters, "K-Means")

# add the cluster labels to the data
kmeans_data_with_categorical = pd.concat([data_with_clusters, RFMD_final['State']], axis=1)

# add the cluster labels to the data
kmeans_data_with_categorical['Cluster'] = labels
data_with_clusters['Cluster'] = labels

# processed data summary
logger.print("K-Means cluster summary (Processed Data):")
logger.print(utils.summarize_cluster(kmeans_data_with_categorical))

# raw data summary
logger.print("K-Means cluster summary (Original Data):")
utils.summarize_cluster_v2(kmeans_rfm_raw)

# evaluation metrics
utils.eval_metrics_single(kmeans_df, labels, logger)

# export results
utils.export_pickle(kmeans_data_with_categorical, "rfmd_kmeans.pkl")
utils.export_pickle(kmeans_rfm_raw, "rfmd_kmeans_clean.pkl")
