from sklearn.cluster import KMeans
from mother import Mother

class KMeansImpl(Mother):
    def __init__(self):
        super().__init__(name="K-Means", polling_interval=0.01)
    
    def _run_clustering(self):
        # param
        n_clusters = 4
        random_state = 42
        # algo
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(self.RFM_numerical)
        return labels
    
if __name__ == "__main__":
    kmeans_impl = KMeansImpl()
    kmeans_impl.run()