from sklearn.cluster import KMeans
from mother import Mother

class KMeansImpl(Mother):
    def __init__(self):
        super().__init__(name="K-Means", polling_interval=0.01)
    
    def _run_clustering(self):
        kmeans = KMeans(
            n_clusters=4, 
            random_state=42
        )
        labels = kmeans.fit_predict(self.RFM_numerical)
        return labels
    
if __name__ == "__main__":
    kmeans_impl = KMeansImpl()
    kmeans_impl.run()