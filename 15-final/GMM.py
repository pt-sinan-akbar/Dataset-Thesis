import pandas as pd
import utils
from sklearn.mixture import GaussianMixture
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

gaussian_df = RFM_numerical.copy()

logger.print("Running GMM clustering")
benchmark.start_benchmark()
gmm = GaussianMixture(n_components=3, random_state=42)
labels = gmm.fit_predict(gaussian_df)
benchmark.end_benchmark()

# Add the cluster labels to the data
data_with_clusters = gaussian_df.copy()
data_with_clusters['Cluster'] = labels

# raw
gmm_rfm_raw = pd.DataFrame(df_clean, columns=["recency", "frequency", "monetary", "State"])
gmm_rfm_raw['Cluster'] = labels
logger.print(gmm_rfm_raw.head())

# Create a 3D scatter plot
utils.plot_3d_clusters(data_with_clusters, "Gaussian Mixture Model")

# add the cluster labels to the data
gmm_data_with_categorical = pd.concat([data_with_clusters, RFMD_final['State']], axis=1)

# add the cluster labels to the data
gmm_data_with_categorical['Cluster'] = labels
data_with_clusters['Cluster'] = labels

# processed data summary
logger.print("GMM cluster summary (Processed Data):")
logger.print(utils.summarize_cluster(gmm_data_with_categorical))

# raw data summary
logger.print("GMM cluster summary (Original Data):")
utils.summarize_cluster_v2(gmm_rfm_raw)

# evaluation metrics
utils.eval_metrics_single(gaussian_df, labels, logger)

# export results
utils.export_pickle(gmm_data_with_categorical, "rfmd_gmm.pkl")
utils.export_pickle(gmm_rfm_raw, "rfmd_gmm_clean.pkl")