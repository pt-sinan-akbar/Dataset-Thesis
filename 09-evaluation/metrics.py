import pickle
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Custom for python script

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

with open('../08-rfmd-final-processing/rfmd_final.pkl', 'rb') as file:
    RFMD_final = pickle.load(file)

with open('../08-rfmd-final-processing/rfm_numerical.pkl', 'rb') as file:
    RFM_numerical = pickle.load(file)

# END

def evaluate_kmeans(X, cluster_range):
    results = {
        'k': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }

    for k in cluster_range:
        # Train K-means model
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(X)
        labels = kmeans.labels_

        # Calculate metrics
        results['k'].append(k)

        # Silhouette Score (higher is better)
        sil_score = silhouette_score(X, labels)
        results['silhouette'].append(sil_score)

        # Davies-Bouldin Score (lower is better)
        db_score = davies_bouldin_score(X, labels)
        results['davies_bouldin'].append(db_score)

        # Calinski-Harabasz Score (higher is better)
        ch_score = calinski_harabasz_score(X, labels)
        results['calinski_harabasz'].append(ch_score)

        print(f"k={k}: Silhouette={sil_score:.4f}, Davies-Bouldin={db_score:.4f}, Calinski-Harabasz={ch_score:.4f}")

    return results

def plot_evaluation_metrics(results):
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot Silhouette Score (higher is better)
    axs[0].plot(results['k'], results['silhouette'], marker='o', linestyle='-', color='blue')
    axs[0].set_title('Silhouette Score vs Number of Clusters (Higher is Better)')
    axs[0].set_xlabel('Number of Clusters (k)')
    axs[0].set_ylabel('Silhouette Score')
    axs[0].grid(True)

    # Plot Davies-Bouldin Score (lower is better)
    axs[1].plot(results['k'], results['davies_bouldin'], marker='o', linestyle='-', color='red')
    axs[1].set_title('Davies-Bouldin Score vs Number of Clusters (Lower is Better)')
    axs[1].set_xlabel('Number of Clusters (k)')
    axs[1].set_ylabel('Davies-Bouldin Score')
    axs[1].grid(True)

    # Plot Calinski-Harabasz Score (higher is better)
    axs[2].plot(results['k'], results['calinski_harabasz'], marker='o', linestyle='-', color='green')
    axs[2].set_title('Calinski-Harabasz Score vs Number of Clusters (Higher is Better)')
    axs[2].set_xlabel('Number of Clusters (k)')
    axs[2].set_ylabel('Calinski-Harabasz Score')
    axs[2].grid(True)

    plt.tight_layout()
    # plt.show()
    plt.savefig('metrics_evaluation.png', dpi=300, bbox_inches='tight')


cluster_range = range(2, 7)
results = evaluate_kmeans(RFM_numerical, cluster_range)
plot_evaluation_metrics(results)