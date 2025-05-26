import pandas as pd
from kmodes.kprototypes import KPrototypes
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

# kproto_df = RFM_numerical.copy()
kproto_df = pd.DataFrame(RFMD_final, columns=['recency', 'frequency', 'monetary', 'State'])
categorical_indices = [3]

logger.print("Running K-Prototype clustering")
benchmark.start_benchmark()
kproto = KPrototypes(n_clusters=4, init='Huang', gamma=1.0, random_state=42)
labels = kproto.fit_predict(kproto_df, categorical=categorical_indices)
benchmark.end_benchmark()

# Add the cluster labels to the data
data_with_clusters = kproto_df.copy()
data_with_clusters['Cluster'] = labels

# raw
kproto_rfm_raw = pd.DataFrame(df_clean, columns=["recency", "frequency", "monetary", "State"])
kproto_rfm_raw['Cluster'] = labels
logger.print(kproto_rfm_raw.head())

# Create a 3D scatter plot
utils.plot_3d_clusters(data_with_clusters, "K-Prototypes")

# add the cluster labels to the data
kproto_data_with_categorical = pd.concat([data_with_clusters, RFMD_final['State']], axis=1)

# add the cluster labels to the data
kproto_data_with_categorical['Cluster'] = labels
data_with_clusters['Cluster'] = labels

# processed data summary
logger.print("K-Prototypes cluster summary (Processed Data):")
logger.print(utils.summarize_cluster(kproto_data_with_categorical))

# raw data summary
logger.print("K-Prototypes cluster summary (Original Data):")
utils.summarize_cluster_v2(kproto_rfm_raw)

# evaluation metrics
utils.eval_metrics_single(kproto_df, labels, logger)

# export results
utils.export_pickle(kproto_data_with_categorical, "rfmd_kproto.pkl")
utils.export_pickle(kproto_rfm_raw, "rfmd_kproto_clean.pkl")
