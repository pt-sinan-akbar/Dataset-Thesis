import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import utils
from utils import evaluation_metrics
from logger import Logger
from benchmark import Benchmark

# Custom for python script
utils.widen_output(pd)
RFMD_final = utils.import_pickle('../08-rfmd-final-processing/rfmd_final.pkl')
RFM_numerical = utils.import_pickle('../08-rfmd-final-processing/rfm_numerical.pkl')
df_clean = utils.import_pickle('../05-outlier/rfmd_clean.pkl')
eval_result = utils.import_pickle('rfm_gmm_eval.pkl')
logger = Logger()
benchmark = Benchmark(logger)
# END

logger.print("Find the optimal number of clusters using the BIC and AIC criteria")

gaussian_df = RFM_numerical.copy()

n_components = range(1, 101)
model = [GaussianMixture(n_components=i, random_state=42).fit(gaussian_df) for i in n_components]

# AICS (Akaike Information Criterion)
aics = [m.aic(gaussian_df) for m in model]

# BICS (Bayesian Information Criterion)
bics = [m.bic(gaussian_df) for m in model]

logger.print("AICs: " + str(aics))
logger.print("BICs: " + str(bics))

logger.print("Plot AIC and BIC")

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
#
# logger.print("Running GMM clustering")
# benchmark.start_benchmark()
# gmm = GaussianMixture(n_components=2, random_state=42)
# labels = gmm.fit_predict(gaussian_df)
# benchmark.end_benchmark()
#
# # Add the cluster labels to the data
# data_with_clusters = gaussian_df.copy()
# data_with_clusters['Cluster'] = labels
#
# # raw
# gmm_rfm_raw = pd.DataFrame(df_clean, columns=["recency", "frequency", "monetary", "State"])
# gmm_rfm_raw['Cluster'] = labels
# logger.print(gmm_rfm_raw.head())
#
# # Create a 3D scatter plot
# utils.plot_3d_clusters(data_with_clusters, "Gaussian Mixture Model")
#
# # add the cluster labels to the data
# gmm_data_with_categorical = pd.concat([data_with_clusters, RFMD_final['State']], axis=1)
#
# # add the cluster labels to the data
# gmm_data_with_categorical['Cluster'] = labels
# data_with_clusters['Cluster'] = labels
#
# # display cluster unique values on cluster with categorical
# logger.print("GMM cluster summary:")
# logger.print(utils.summarize_cluster(gmm_data_with_categorical))
# #
# # cluster summary with raw data
# logger.print("GMM cluster summary (Original Data):")
# logger.print(utils.summarize_cluster(gmm_rfm_raw, False))
#
# utils.summarize_cluster_v2(gmm_rfm_raw)
#
# logger.print("evaluation metrics")
# eval_results = evaluation_metrics(
#     df=gmm_rfm_raw,
#     algorithm="GMM",
#     cluster_range=range(2, 7)
# )
#
# logger.print("Evaluation results:")
# logger.print(eval_results)
#
# logger.print("Plot evaluation metrics")
# utils.plot_evaluation_metrics(eval_results)
#
# utils.export_pickle(gmm_data_with_categorical, "rfmd_gmm.pkl")
# utils.export_pickle(gmm_rfm_raw, "rfmd_gmm_clean.pkl")
# utils.export_pickle(eval_results, "rfm_gmm_eval.pkl")
#


