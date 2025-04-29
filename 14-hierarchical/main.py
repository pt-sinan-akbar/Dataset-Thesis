import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import argparse
import utils


class Hierarchical(object):
    def __init__(self):
        self.rfm_hierarchical_clean = None
        self.rfm_hierarchical = None
        self.rfm_numerical = utils.import_pickle('../08-rfmd-final-processing/rfm_numerical.pkl')
        self.df_clean = utils.import_pickle('../05-outlier/rfmd_clean.pkl')
        utils.widen_output(pd)

    def run_dendogram(self):
        print("--- Dendogram ---")

        # Calculate the linkage matrix
        # 'ward' minimizes the variance within each cluster, requires Euclidean distance
        linked = linkage(self.rfm_numerical, method='ward', metric='euclidean')

        # Plot dendogram
        plt.figure(figsize=(12, 7))
        dendrogram(linked,
                   orientation='top',
                   distance_sort='descending',
                   show_leaf_counts=True)
        plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
        plt.xlabel('Data Points (or Index)')
        plt.ylabel('Euclidean Distance (Ward)')
        plt.axhline(y=2.5, color='r', linestyle='--', label='Example Cutoff')
        plt.legend()
        plt.suptitle(
            "Look for the largest vertical distance without crossing horizontal lines to suggest k, or cut at a specific distance.")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('dendrogram_hierarchical.png', dpi=300, bbox_inches='tight')
        print("Dendogram saved to file")

    def agglomerative(self, chosen_k_hierarchical: int):
        print(f"--- Agglomerative Clustering with k = {chosen_k_hierarchical} ---")

        # Choose k based on dendrogram inspection (e.g., cutting where y=2.5 gives k=3)
        agg_clust = AgglomerativeClustering(n_clusters=chosen_k_hierarchical, metric='euclidean', linkage='ward')

        # Fit and get cluster labels
        hierarchical_labels = agg_clust.fit_predict(self.rfm_numerical)

        # Add labels to your data
        rfm_hierarchical = self.rfm_numerical.copy()
        rfm_hierarchical['Cluster'] = hierarchical_labels
        print(f"Cluster counts:\n{rfm_hierarchical['Cluster_Hierarchical'].value_counts()}")
        print(rfm_hierarchical.head())

        rfm_hierarchical_clean = self.df_clean.copy()
        rfm_hierarchical_clean['Cluster'] = hierarchical_labels
        print("Clean data sanity test:")
        print("raw rfm count:", len(self.df_clean))
        print("hierarchical count:", len(hierarchical_labels))
        print(rfm_hierarchical_clean.head())

        rfm_hierarchical.to_pickle('rfm_hierarchical.pkl')
        rfm_hierarchical_clean.to_pickle('rfm_hierarchical_clean')
        self.rfm_hierarchical = rfm_hierarchical
        self.rfm_hierarchical_clean = rfm_hierarchical_clean

    def summary(self):
        print("--- Summary ---")
        if not self.rfm_hierarchical or not self.rfm_hierarchical_clean:
            raise ValueError("No hierarchical clustering data available. Please run agglomerative first.")

        # summary
        print("Cluster summary:")
        print(utils.summarize_cluster(self.rfm_hierarchical))

        print("Cluster summary (Original Data):")
        print(utils.summarize_cluster(self.rfm_hierarchical_clean, False))

        # plot 3d clusters
        utils.plot_3d_clusters(self.rfm_hierarchical_clean, "Hierarchical")

    def export_result(self):
        print("--- Exporting Result ---")
        self.rfm_hierarchical_clean.to_pickle('rfm_hierarchical_clean.pkl')
        self.rfm_hierarchical.to_pickle('rfm_hierarchical.pkl')
        print("Exported to file")


if __name__ == "__main__":
    hierarchical = Hierarchical()

    # check args
    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_k')
    cl_args = parser.parse_args()
    chosen_k = cl_args.chosen_k

    # gow
    if not chosen_k:
        print("No K in args, will run dendogram only")
        hierarchical.run_dendogram()
    else:
        if not isinstance(chosen_k, int):
            raise ValueError("chosen_k_hierarchical must be an integer.")
        else:
            hierarchical.run_dendogram()
            hierarchical.agglomerative(chosen_k)
            hierarchical.summary()
            hierarchical.export_result()
