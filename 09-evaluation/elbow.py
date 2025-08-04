import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Custom for python script

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

with open('../08-rfmd-final-processing/rfmd_final.pkl', 'rb') as file:
    RFMD_final = pickle.load(file)

# END

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
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(RFM_numerical)
    inertias.append(kmeans.inertia_)
    print("WCSS for k =", k, "is", kmeans.inertia_)

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

plt.cla()

plt.figure(figsize=(10, 6))

plt.plot(cluster_range, inertias, marker='o', linestyle='-', color='b')
# plt.plot(cluster_range, wcss, marker='s', linestyle='--', color='r', label='RFMD')

plt.title('Elbow Method for K-Means')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.xticks(cluster_range)
plt.grid(True)
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig('elbow_kmeans.png', dpi=300, bbox_inches='tight')

wcss_diffs = np.diff(inertias)
percentage_decrease = [-diff/inertias[i] * 100 for i, diff in enumerate(wcss_diffs)]

print("\nPercentage difference between consecutive K values:")
for i, decrease in enumerate(percentage_decrease):
    print(f"From K={i+1} to K={i+2}: {decrease:.2f}% decrease")