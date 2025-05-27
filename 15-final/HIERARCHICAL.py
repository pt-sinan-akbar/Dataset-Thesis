from sklearn.cluster import AgglomerativeClustering
from mother import Mother

class Hierarchical(Mother):
    def __init__(self):
        super().__init__(name="Hierarchical", polling_interval=3)
    
    def _run_clustering(self):
        hierarchical = AgglomerativeClustering(
            n_clusters=4, 
            metric='euclidean', 
            linkage='ward'
        )
        labels = hierarchical.fit_predict(self.RFM_numerical)
        return labels

if __name__ == "__main__":
    hierarchical_impl = Hierarchical()
    hierarchical_impl.run()