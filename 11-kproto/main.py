import pandas as pd
from kmodes.kprototypes import KPrototypes
import utils
from logger import Logger
from benchmark import Benchmark

# Custom for python script
utils.widen_output(pd)

df_clean = utils.import_pickle('../05-outlier/rfmd_clean.pkl')
encoded_to_state = utils.import_pickle('../08-rfmd-final-processing/encoded_to_state.pkl')
state_mapping = utils.import_pickle('../08-rfmd-final-processing/state_mapping.pkl')
RFMD_final = utils.import_pickle('../08-rfmd-final-processing/rfmd_final.pkl')
logger = Logger()
benchmark = Benchmark(logger)
# END

# Setelah dapet cluster yg bagus dari atas
best_k = 4

# imported from huang
rfmd_kproto = pd.DataFrame(RFMD_final, columns=['recency', 'frequency', 'monetary', 'State'])
gamma_value = 1.0
categorical_indices = [3]
# end
logger.print("Running K-Prototypes")
benchmark.start_benchmark()
kproto = KPrototypes(n_clusters=best_k, init='Huang', gamma=gamma_value)
clusters_kproto = kproto.fit_predict(rfmd_kproto, categorical=categorical_indices)
benchmark.end_benchmark()
rfmd_kproto['Cluster'] = clusters_kproto

logger.print(rfmd_kproto.head())

logger.print("raw rfmd count:", len(df_clean))
logger.print("kproto cluster count:", len(kproto.labels_))
kproto_rfmd_raw = pd.DataFrame(df_clean, columns=["recency", "frequency", "monetary", "State"])
kproto_rfmd_raw['Cluster'] = kproto.labels_
logger.print(kproto_rfmd_raw.head())

# summary
logger.print("Cluster summary:")
logger.print(utils.summarize_cluster(rfmd_kproto))

# logger.print("KProto cluster summary (Original Data):")
logger.print("Cluster summary (Original Data):")
logger.print(utils.summarize_cluster(kproto_rfmd_raw, False))

# plot 3d clusters
utils.plot_3d_clusters(kproto_rfmd_raw, "K-Prototypes")

utils.summarize_cluster_v2(kproto_rfmd_raw)


logger.print("evaluation metrics")
eval_results = utils.evaluation_metrics(
    df=rfmd_kproto,
    algorithm="KPrototypes",
    cluster_range=range(2, 7),
)
logger.print("Evaluation results:")
logger.print(eval_results)

logger.print("Plot evaluation metrics")
utils.plot_evaluation_metrics(eval_results)


utils.export_pickle(rfmd_kproto, "rfmd_kproto.pkl")
utils.export_pickle(kproto_rfmd_raw, "rfmd_kproto_clean.pkl")
