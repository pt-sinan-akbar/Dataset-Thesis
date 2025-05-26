import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
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

hierarchical_df = RFM_numerical.copy()

logger.print("Running Hierarchical clustering")
benchmark.start_benchmark()
hierarchical = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
labels = hierarchical.fit_predict(hierarchical_df)
benchmark.end_benchmark()

# Add the cluster labels to the data
data_with_clusters = hierarchical_df.copy()
data_with_clusters['Cluster'] = labels

# raw
hierarchical_rfm_raw = pd.DataFrame(df_clean, columns=["recency", "frequency", "monetary", "State"])
hierarchical_rfm_raw['Cluster'] = labels
logger.print(hierarchical_rfm_raw.head())

# Create a 3D scatter plot
utils.plot_3d_clusters(data_with_clusters, "Hierarchical")

# add the cluster labels to the data
hierarchical_data_with_categorical = pd.concat([data_with_clusters, RFMD_final['State']], axis=1)

# add the cluster labels to the data
hierarchical_data_with_categorical['Cluster'] = labels
data_with_clusters['Cluster'] = labels

# processed data summary
logger.print("Hierarchical cluster summary (Processed Data):")
logger.print(utils.summarize_cluster(hierarchical_data_with_categorical))

# raw data summary
logger.print("Hierarchical cluster summary (Original Data):")
utils.summarize_cluster_v2(hierarchical_rfm_raw)

# evaluation metrics
utils.eval_metrics_single(hierarchical_df, labels, logger)

# export results
utils.export_pickle(hierarchical_data_with_categorical, "rfmd_hierarchical.pkl")
utils.export_pickle(hierarchical_rfm_raw, "rfmd_hierarchical_clean.pkl")