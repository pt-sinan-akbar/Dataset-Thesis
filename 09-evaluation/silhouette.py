import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import cm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utils import plot_3d_clusters
import pickle

with open('../08-rfmd-final-processing/rfmd_final.pkl', 'rb') as file:
    RFMD_final = pickle.load(file)

with open('../08-rfmd-final-processing/rfm_numerical.pkl', 'rb') as file:
    RFM_numerical = pickle.load(file)

def silhouette_analysis(X, cluster_range=range(2, 11)):
    silhouette_scores = []
    for k in cluster_range:
        # print the current k
        print(f"Evaluating k={k}")

        # prevent k from being greater than the number of samples
        if k >= len(X):
            print(f"Skipping k={k} because it's greater than or equal to the number of samples")
            continue

        # do KMeans clustering
        kmeans = KMeans(n_clusters=k, n_init=10)
        labels = kmeans.fit_predict(X)

        # calculate silhouette score 
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

        # print the silhouette score
        sample_silhouette_values = silhouette_samples(X, labels)

        # plot the silhouette score
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))  # Only one subplot

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (k + 1) * 10])
        y_lower = 10

        for i in range(k):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / k)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax1.set_title('Silhouette Plot for Various Clusters')
        ax1.set_xlabel('Silhouette Coefficient Values')
        ax1.set_ylabel('Cluster Label')
        ax1.axvline(x=score, color='red', linestyle='--')
        ax1.set_yticks([])

        # plt.show()
        plt.savefig(f'silhouette_plot_{k}.png', dpi=300, bbox_inches='tight')

        # **3D Cluster Visualization**
        # concat state to the dataframe
        X_with_labels = X.copy()
        X_with_labels = pd.DataFrame(X_with_labels, columns=['recency', 'frequency', 'monetary'])
        X_with_labels = pd.concat([X_with_labels, RFMD_final['State']], axis=1)
        X_with_labels['Cluster'] = labels


        plot_3d_clusters(X_with_labels, algorithm_name=f"silhouette_k_means(k={k})")

    # **Final Silhouette Score Plot**
    plt.figure(figsize=(8, 6))
    plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-')
    plt.title('Silhouette Method for Optimal Number of Clusters (k)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    # plt.show()
    plt.savefig('silhouette_method.png', dpi=300, bbox_inches='tight')

    return silhouette_scores

X = RFM_numerical.copy()

# do a sillhouette score to evaluate the model
silhouette_scores = silhouette_analysis(X, range(2, 7))