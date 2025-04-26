import pickle
import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt

with open('../08-rfmd-final-processing/rfmd_final.pkl', 'rb') as file:
    RFMD_final = pickle.load(file)

# select the features for K-
rfmd_kproto = pd.DataFrame(RFMD_final, columns=['recency', 'frequency', 'monetary', 'State'])
numerical_indices = [0, 1, 2]
categorical_indices = [3]

def huang_cost_function(X, centroids, clusters, categorical_indices, gamma=1.0):
    n_samples = X.shape[0]
    cost = 0.0

    numerical_indices = [i for i in range(X.shape[1]) if i not in categorical_indices]

    for i in range(n_samples):
        cluster_idx = clusters[i]
        centroid = centroids[cluster_idx]

        numerical_cost = 0
        for j in numerical_indices:
            numerical_cost += (X[i, j] - centroid[j]) ** 2

        categorical_cost = 0
        for j in categorical_indices:
            if X[i, j] != centroid[j]:
                categorical_cost += 1

        # Total cost with gamma weight
        cost += numerical_cost + gamma * categorical_cost

    return cost

def elbow_method_with_huang(data, max_clusters=10, gamma=1.0):
    costs = []
    X = data.values

    for k in range(1, max_clusters + 1):
        try:
            kproto = KPrototypes(n_clusters=k, init='Huang', gamma=gamma, random_state=None, verbose=0)
            clusters = kproto.fit_predict(X, categorical=categorical_indices)
            centroids = kproto.cluster_centroids_

            cost = huang_cost_function(X, centroids, clusters, categorical_indices, gamma)
            costs.append(cost)
            print(f"K = {k}, Huang Cost = {cost}")
        except Exception as e:
            print(f"Error for k={k}: {e}")
            if costs:
                costs.append(costs[-1])
            else:
                costs.append(float('inf'))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), costs, marker='o', linestyle='-')
    plt.title('Elbow Method with Huang Cost Function')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Huang Cost Function')
    plt.xticks(range(1, max_clusters + 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('elbow_curve.png')
    plt.show()

    for i, cost in enumerate(costs):
        print(f"K={i+1}: Cost={cost}")

    cost_diffs = np.diff(costs)
    percentage_decrease = [-diff/costs[i] * 100 for i, diff in enumerate(cost_diffs)]

    for i, decrease in enumerate(percentage_decrease):
        print(f"From K={i+1} to K={i+2}: {decrease:.2f}% decrease")

    return costs

gamma_value = 1.0
# costs = elbow_method_with_huang(rfmd_kproto, max_clusters=10, gamma=gamma_value)

# Karena elbow RFMD dengan huang cost ini memakan waktu lama, maka akan digunakan data dari run sebelumnya

costs = [302945.9999999998, 200403.4673796627, 144504.98638342315, 111923.87470059782, 98687.1746289738,
         90285.75967918534, 82880.50360656573, 74927.28467282065, 70738.93183310021, 66556.70413009671]

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), costs, marker='o', linestyle='-')
plt.title('Elbow Method with Huang Cost Function')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Huang Cost Function')
plt.xticks(range(1, 11))
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig('huang_cost_rfmd.png', dpi=300, bbox_inches='tight')

rfmd_kproto.to_pickle('rfmd_kproto.pkl')