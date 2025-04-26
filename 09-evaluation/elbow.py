import pickle
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

with open('../08-rfmd-final-processing/rfmd_final.pkl', 'rb') as file:
    RFMD_final = pickle.load(file)

# Elbow method RFMD

#ilangin column Code
RFMD_numerical = RFMD_final.drop(columns=['Code'])

wcss = []
cluster_range = range(1, 11)
for i in cluster_range:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(RFMD_numerical)
    wcss.append(kmeans.inertia_)

# Elbow method RFM

RFM_numerical = pd.DataFrame(RFMD_final, columns=['recency', 'frequency', 'monetary'])
inertias = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(RFM_numerical)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))

plt.plot(cluster_range, inertias, marker='o', linestyle='-', color='b', label='RFM')
plt.plot(cluster_range, wcss, marker='s', linestyle='--', color='r', label='RFMD')

plt.title('Elbow Method Comparison')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia / WCSS')
plt.xticks(cluster_range)
plt.grid(True)
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig('elbow_both.png', dpi=300, bbox_inches='tight')