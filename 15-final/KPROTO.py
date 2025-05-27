from kmodes.kprototypes import KPrototypes
from mother import Mother

class KProto(Mother):
    def __init__(self):
        super().__init__(name="K-Prototype", polling_interval=1, is_rfmd=True)
    
    def _run_clustering(self):
        # using RFMD, need to be customized here
        kproto_df = self.RFM_numerical.copy()
        kproto_df['State'] = self.RFM_categorical
        # algo
        kproto = KPrototypes(
            n_clusters=4, 
            init='Huang', 
            gamma=1.0, 
            random_state=42
        )
        labels = kproto.fit_predict(
            kproto_df, 
            categorical=[3]
        )
        return labels

if __name__ == "__main__":
    kproto_impl = KProto()
    kproto_impl.run()
