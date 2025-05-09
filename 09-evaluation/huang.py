from datetime import datetime
import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
import argparse
import utils

# Custom for python script
utils.widen_output(pd)
RFMD_final = utils.import_pickle('../08-rfmd-final-processing/rfmd_final.pkl')
# END

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

def elbow_method_with_huang(data, max_clusters=10, gamma=1.0, costs=None):
    if not costs:
        costs = []
        X = data.values
    
        for k in range(1, max_clusters + 1):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing K={k}...")
            try:
                kproto = KPrototypes(n_clusters=k, init='Huang', gamma=gamma, random_state=None, verbose=0)
                clusters = kproto.fit_predict(X, categorical=categorical_indices)
                centroids = kproto.cluster_centroids_
    
                cost = huang_cost_function(X, centroids, clusters, categorical_indices, gamma)
                costs.append(cost)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] K = {k}, Huang Cost = {cost}")
            except Exception as e:
                print(f"Error for k={k}: {e}")
                if costs:
                    costs.append(costs[-1])
                else:
                    costs.append(float('inf'))
    else:
        print("Using precomputed costs...")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), costs, marker='o', linestyle='-')
    plt.title('Elbow Method with Huang Cost Function')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Huang Cost Function')
    plt.xticks(range(1, max_clusters + 1))
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('huang_elbow_curve.png')

    for i, cost in enumerate(costs):
        print(f"K={i+1}: Cost={cost}")

    cost_diffs = np.diff(costs)
    percentage_decrease = [-diff/costs[i] * 100 for i, diff in enumerate(cost_diffs)]

    print("\nPercentage difference between consecutive K values:")
    for i, decrease in enumerate(percentage_decrease):
        print(f"From K={i+1} to K={i+2}: {decrease:.2f}% decrease")

    return costs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_huang', type=bool, default=False)
    cl_args = parser.parse_args()
    run_huang = cl_args.run_huang

    # select the features for K-
    rfmd_kproto = pd.DataFrame(RFMD_final, columns=['recency', 'frequency', 'monetary', 'State'])
    numerical_indices = [0, 1, 2]
    categorical_indices = [3]
    gamma_value = 1.0

    if run_huang:
        print("Running Huang Elbow Method...")
        elbow_method_with_huang(rfmd_kproto, max_clusters=10, gamma=gamma_value)
    else:
        # Karena elbow RFMD dengan huang cost ini memakan waktu lama, dapat digunakan data dari run sebelumnya
        costs_precomputed = [302947.9999999984, 200397.5126547657, 144499.1736882579, 111856.6571584246, 98594.2439447426,
                             90201.61211915832, 82231.18385391243, 74867.38920003947, 70673.36812059133, 66476.8060219641]
        elbow_method_with_huang(rfmd_kproto, max_clusters=10, gamma=gamma_value, costs=costs_precomputed)
