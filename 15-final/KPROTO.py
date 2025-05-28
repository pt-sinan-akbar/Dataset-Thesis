from kmodes.kprototypes import KPrototypes

import utils
from mother import Mother

class KProto(Mother):
    def __init__(self):
        super().__init__(name="K-Prototype", polling_interval=1, is_rfmd=True)
        # using RFMD, customized here
        self.kproto_df = self.RFM_numerical.copy()
        self.kproto_df['State'] = self.RFM_categorical
    
    def _run_clustering(self):
        # algo
        kproto = KPrototypes(
            n_clusters=4, 
            init='Huang', 
            gamma=1.0, 
            random_state=42,
            n_jobs=-1
        )
        labels = kproto.fit_predict(
            self.kproto_df, 
            categorical=[3]
        )
        return labels
    
    def _run_eval_metrics(self, labels):
        utils.eval_metrics_single(self.kproto_df, labels, self.logger)

if __name__ == "__main__":
    kproto_impl = KProto()
    kproto_impl.run()
