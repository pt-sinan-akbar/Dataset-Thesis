from sklearn.cluster import DBSCAN
import utils
from mother import Mother
from copy import deepcopy

class DBSCANImpl(Mother):
    def __init__(self):
        super().__init__(name="DBSCAN", polling_interval=0.1)

    def _run_clustering(self):
        dbscan = DBSCAN(
            eps=0.199, 
            min_samples=11, 
            metric='manhattan', 
            n_jobs=-1
        )
        labels = dbscan.fit_predict(self.RFM_numerical)
        return labels

    def _run_eval_metrics(self, labels):
        numerical_df_copy = self.RFM_numerical.copy()
        labels_copy = deepcopy(labels)
        filtered_indices = labels_copy != -1
        filtered_labels = labels_copy[filtered_indices]
        filtered_dbscan_df = numerical_df_copy[filtered_indices]
        utils.eval_metrics_single(filtered_dbscan_df, filtered_labels, self.logger)
        
if __name__ == "__main__":
    dbscan_impl = DBSCANImpl()
    dbscan_impl.run()