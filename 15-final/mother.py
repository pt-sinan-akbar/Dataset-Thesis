import pandas as pd
import utils
from logger import Logger
from benchmark import Benchmark

class Mother(object):
    """
    Mother class for clustering algorithms.
    This class is used to run clustering algorithms on the RFM data.
    It is designed to be inherited by specific clustering algorithm classes.
    """
    
    def __init__(self, name, polling_interval=0.1):
        """
        Initialize the Mother class.
        This method sets up the necessary imports and configurations for clustering.
        """
        # Custom for python script
        utils.widen_output(pd)
        self.RFM_categorical = utils.import_pickle('../08-rfmd-final-processing/rfm_categorical.pkl')
        self.RFM_numerical = utils.import_pickle('../08-rfmd-final-processing/rfm_numerical.pkl')
        self.df_clean = utils.import_pickle('../05-outlier/rfmd_clean.pkl')
        self.logger = Logger()
        self.benchmark = Benchmark(logger=self.logger, polling_interval=polling_interval)
        self.name = name
        self.shortname = name.replace('-', '').lower()

    def _run_clustering(self):
        """
        Run the clustering algorithm.
        This method should be overridden by subclasses to implement specific clustering algorithms.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def run(self):
        algo_df = self.RFM_numerical.copy()
        
        self.logger.print(f"Running {self.name} clustering")
        self.benchmark.start_benchmark()
        labels = self._run_clustering()
        self.benchmark.end_benchmark()
        
        # Check labels len for sanity test
        if len(labels) != len(self.df_clean) or len(labels) != len(self.RFM_numerical) or len(labels) != len(self.RFM_categorical):
            raise ValueError("The number of labels does not match the number of data points.")
        else:
            self.logger.print(f"{self.name} clustering completed with {len(set(labels))} clusters consisting of {len(labels)} data points.")
        
        # Add the cluster labels to the data and enrich with State
        algo_df['State'] = self.RFM_categorical
        algo_df['Cluster'] = labels
        self.logger.print(f"{self.name} clustering data sample:")
        self.logger.print(algo_df.head())
        
        # Add cluster labels to the original data and enrich with State
        algo_df_raw = pd.DataFrame(self.df_clean, columns=["recency", "frequency", "monetary", "State"])
        algo_df_raw['Cluster'] = labels
        self.logger.print(f"{self.name} clustering original data sample:")
        self.logger.print(algo_df_raw.head())
        
        # summary
        self.logger.print(f"{self.name} cluster summary:")
        self.logger.print(utils.summarize_cluster(algo_df))
        utils.plot_3d_clusters(algo_df, f"{self.name}")
        utils.plot_pca(algo_df, labels)
        
        # original data summary
        self.logger.print(f"{self.name} cluster summary (Original Data):")
        utils.summarize_cluster_v2(algo_df_raw)
        utils.plot_3d_clusters(algo_df_raw, f"{self.name} (Clean)")
        utils.plot_pca(algo_df_raw, labels)
        
        # evaluation metrics
        utils.eval_metrics_single(self.RFM_numerical, labels, self.logger)
        
        # export results
        utils.export_pickle(algo_df, f"{self.shortname}_result.pkl")
        utils.export_pickle(algo_df_raw, f"{self.shortname}_clean_result.pkl")
