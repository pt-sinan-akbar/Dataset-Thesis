import pandas as pd
from sklearn.cluster import DBSCAN
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

dbscan_df = RFM_numerical.copy()

logger.print("Running DBSCAN clustering")
benchmark.start_benchmark()
dbscan = DBSCAN(eps=0.1990, min_samples=11, metric='manhattan', n_jobs=-1)
labels = dbscan.fit_predict(dbscan_df)
benchmark.end_benchmark()

# Add the cluster labels to the data
data_with_clusters = dbscan_df.copy()
data_with_clusters['Cluster'] = labels

# raw
dbscan_rfm_raw = pd.DataFrame(df_clean, columns=["recency", "frequency", "monetary", "State"])
dbscan_rfm_raw['Cluster'] = labels
logger.print(dbscan_rfm_raw.head())

# Create a 3D scatter plot
utils.plot_3d_clusters(data_with_clusters, "DBSCAN")

# add the cluster labels to the data
dbscan_data_with_categorical = pd.concat([data_with_clusters, RFMD_final['State']], axis=1)

# processed data summary
logger.print("DBSCAN cluster summary (Processed Data):")
logger.print(utils.summarize_cluster(dbscan_data_with_categorical))

# raw data summary
logger.print("DBSCAN cluster summary (Original Data):")
utils.summarize_cluster_v2(dbscan_rfm_raw)

# evaluation metrics
# filter out noise points
filtered_indices = labels != -1
filtered_labels = labels[filtered_indices]
filtered_dbscan_df = dbscan_df[filtered_indices]
utils.eval_metrics_single(filtered_dbscan_df, filtered_labels, logger)

# export results
utils.export_pickle(dbscan_data_with_categorical, 'dbscan_data_with_categorical.pkl')
utils.export_pickle(dbscan_rfm_raw, 'dbscan_rfm_raw.pkl')




