# iterate trough 1-100, and then plot the KNN for each k
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import utils

RFMD_final = utils.import_pickle('../08-rfmd-final-processing/rfmd_final.pkl')
df_clean = utils.import_pickle('../05-outlier/rfmd_clean.pkl')
state_mapping = utils.import_pickle('../08-rfmd-final-processing/state_mapping.pkl')

X = RFMD_final.drop(columns=['Code', 'State'])
numeric_df = RFMD_final.select_dtypes(include=['float64', 'int64'])
# Define parameter grid (efficient version)
min_samples_values = np.arange(5, 101, 5)  # 5, 10, 15, ..., 100
eps_values = np.arange(0.1, 1.01, 0.05)  # 0.1, 0.15, 0.2, ..., 1.0

best_score = -1
best_params = {}

for eps in eps_values:
    for min_samples in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)

        # Only compute silhouette_score if there are at least 2 clusters and not all noise
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1 and n_clusters < len(X):
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}

print("Best parameters:", best_params)
print("Best silhouette score:", best_score)