from sklearn.mixture import GaussianMixture
from mother import Mother

class GMM(Mother):
    def __init__(self):
        super().__init__(name="GMM", polling_interval=0.01)
        
    def _run_clustering(self):
        gmm = GaussianMixture(
            n_components=3, 
            random_state=42
        )
        labels = gmm.fit_predict(self.RFM_numerical)
        return labels
    
if __name__ == "__main__":
    gmm_impl = GMM()
    gmm_impl.run()