import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import utils

# Custom for python script
utils.widen_output(pd)
RFMD_final = utils.import_pickle('../08-rfmd-final-processing/rfmd_final.pkl')
df_clean = utils.import_pickle('../05-outlier/rfmd_clean.pkl')
state_mapping = utils.import_pickle('../08-rfmd-final-processing/state_mapping.pkl')
# END

#K-Distance Plot buat nyari nilai optimal eps

# We'll use the same X you plan to feed into DBSCAN
X = RFMD_final.drop(columns=['Code']).values

# Set k = min_samples (usually 4 or 5)
k = 5
neigh = NearestNeighbors(n_neighbors=k)
nbrs = neigh.fit(X)

distances, indices = nbrs.kneighbors(X)
k_distances = distances[:, k-1]

# Sort distances for plotting
k_distances = np.sort(k_distances)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.title(f'k-distance Graph (k={k})')
plt.xlabel('Data Points sorted by distance')
plt.ylabel(f'Distance to {k}th Nearest Neighbor')
plt.grid(True)
# plt.show()
plt.savefig('k_distance_graph.png', dpi=300, bbox_inches='tight')

#test
def test_dbscan_eps(RFMD_df, eps_values, min_samples=5):
    results = []

    # Remove categorical columns for DBSCAN
    X = RFMD_df.drop(columns=['Code']).values

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        # Count unique clusters (excluding noise)
        unique_clusters = set(labels)
        num_clusters = len(unique_clusters - {-1})
        noise_ratio = (labels == -1).sum() / len(labels)

        if num_clusters > 1:
            filtered_X = X[labels != -1]
            filtered_labels = labels[labels != -1]
            silhouette = silhouette_score(filtered_X, filtered_labels)
        else:
            silhouette = None  # not valid if only 1 cluster

        results.append({
            'eps': eps,
            'Clusters': num_clusters,
            'Noise Ratio': round(noise_ratio, 3),
            'Silhouette Score': round(silhouette, 4) if silhouette is not None else "N/A"
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)

    # Optional plot
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['eps'],
             [s if s != "N/A" else 0 for s in results_df['Silhouette Score']],
             marker='o', label='Silhouette Score')
    plt.plot(results_df['eps'], results_df['Noise Ratio'], marker='x', label='Noise Ratio')
    plt.title("DBSCAN Evaluation across Eps values")
    plt.xlabel("Eps Value")
    plt.ylabel("Score / Ratio")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('dbscan_eps_eval.png', dpi=300, bbox_inches='tight')

    return results_df

#>0.5 IS NOISE (NOT OPTIMAL)
eps_range = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
result_table = test_dbscan_eps(RFMD_final, eps_range, min_samples=5)

# Evaluation metrics

def evaluate_dbscan(RFMD_final, eps_values, min_samples=5):
    results = {
        'eps': [],
        'num_clusters': [],
        'noise_ratio': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }

    # X_numeric = RFMD_final.drop(columns=['Code']).select_dtypes(include=[np.number]).values
    X_numeric = pd.DataFrame(RFMD_final, columns=['recency', 'frequency', 'monetary'])

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_numeric)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = np.mean(labels == -1)

        results['eps'].append(eps)
        results['num_clusters'].append(num_clusters)
        results['noise_ratio'].append(round(noise_ratio, 3))

        if num_clusters > 1:
            core_mask = labels != -1
            silhouette = silhouette_score(X_numeric[core_mask], labels[core_mask])
            db_score = davies_bouldin_score(X_numeric[core_mask], labels[core_mask])
            ch_score = calinski_harabasz_score(X_numeric[core_mask], labels[core_mask])
        else:
            silhouette = np.nan
            db_score = np.nan
            ch_score = np.nan

        results['silhouette'].append(round(silhouette, 4) if silhouette is not np.nan else np.nan)
        results['davies_bouldin'].append(round(db_score, 4) if db_score is not np.nan else np.nan)
        results['calinski_harabasz'].append(round(ch_score, 4) if ch_score is not np.nan else np.nan)

    return pd.DataFrame(results)

def plot_dbscan_evaluation(results_df):
    fig, axs = plt.subplots(5, 1, figsize=(12, 25))

    axs[0].plot(results_df['eps'], results_df['num_clusters'], marker='o')
    axs[0].set_title('Number of Clusters vs Eps')
    axs[0].set_xlabel('Eps')
    axs[0].set_ylabel('Number of Clusters')
    axs[0].grid(True)

    axs[1].plot(results_df['eps'], results_df['noise_ratio'], marker='x', color='purple')
    axs[1].set_title('Noise Ratio vs Eps')
    axs[1].set_xlabel('Eps')
    axs[1].set_ylabel('Noise Ratio')
    axs[1].grid(True)

    axs[2].plot(results_df['eps'], [v if not np.isnan(v) else 0 for v in results_df['silhouette']], marker='o', color='blue')
    axs[2].set_title('Silhouette Score (Higher is Better)')
    axs[2].set_xlabel('Eps')
    axs[2].set_ylabel('Silhouette Score')
    axs[2].grid(True)

    axs[3].plot(results_df['eps'], [v if not np.isnan(v) else 0 for v in results_df['davies_bouldin']], marker='o', color='red')
    axs[3].set_title('Davies-Bouldin Score (Lower is Better)')
    axs[3].set_xlabel('Eps')
    axs[3].set_ylabel('Davies-Bouldin')
    axs[3].grid(True)

    axs[4].plot(results_df['eps'], [v if not np.isnan(v) else 0 for v in results_df['calinski_harabasz']], marker='o', color='green')
    axs[4].set_title('Calinski-Harabasz Score (Higher is Better)')
    axs[4].set_xlabel('Eps')
    axs[4].set_ylabel('Calinski-Harabasz')
    axs[4].grid(True)

    plt.tight_layout()
    # plt.show()
    plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')

eps_range = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
dbscan_eval_results = evaluate_dbscan(RFMD_final, eps_range)
plot_dbscan_evaluation(dbscan_eval_results)

#DBSCAN COBA
# Berdasarkan Curve diatas 0.35 adalah yang paling bagus
X = RFMD_final.drop(columns=['Code']).values

dbscan = DBSCAN(eps=0.35, min_samples=5)

#Fit dan prediksi klusternya
dbscan_labels = dbscan.fit_predict(X)

RFMD_final['DBSCAN_Cluster'] = dbscan_labels

if len(set(dbscan_labels)) > 1:
    silhouette_avg = silhouette_score(X, dbscan_labels)
    print(f"Silhouette Score : {silhouette_avg}")
else:
    print("DBSCAN Resulted in a single cluster or noise.")

print(RFMD_final.head())


#plot
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', s=50, alpha=0.7)
plt.title("DBSCAN Clustering")
plt.xlabel("Recency")
plt.ylabel("Frequency")
plt.colorbar(label="Cluster")
# plt.show()
plt.savefig('dbscan_cluster.png', dpi=300, bbox_inches='tight')

#wow
filtered_data = RFMD_final[RFMD_final['DBSCAN_Cluster'] != -1]
print(f"Filtered data shape (excluding noise): {filtered_data.shape}")

plt.figure(figsize=(10, 6))
plt.scatter(filtered_data['recency'], filtered_data['frequency'],
            c=filtered_data['DBSCAN_Cluster'], cmap='viridis', s=50, alpha=0.7)

# Add labels and title
plt.title("Filtered Data (DBSCAN Clustering)")
plt.xlabel("Recency")
plt.ylabel("Frequency")
plt.colorbar(label="Cluster")
# plt.show()
plt.savefig('dbscan_filtered_data.png', dpi=300, bbox_inches='tight')

noise = RFMD_final[RFMD_final['DBSCAN_Cluster'] == -1]
clusters = RFMD_final[RFMD_final['DBSCAN_Cluster'] != -1]

plt.figure(figsize=(10, 6))
plt.scatter(clusters['recency'], clusters['frequency'],
            c=clusters['DBSCAN_Cluster'], cmap='viridis', s=50, alpha=0.7, label='Clusters')
plt.scatter(noise['recency'], noise['frequency'],
            c='red', s=30, alpha=0.5, label='Noise')

plt.title("DBSCAN Clustering (With Noise in Red)")
plt.xlabel("Recency")
plt.ylabel("Frequency")
plt.legend()
plt.colorbar(label="Cluster")
# plt.show()
plt.savefig('dbscan_noise.png', dpi=300, bbox_inches='tight')

DBSCAN_df = RFMD_final.copy()

# rename the DBSCAN_Cluster to Cluster
DBSCAN_df.rename(columns={'DBSCAN_Cluster': 'Cluster'}, inplace=True)

# drop the code column
DBSCAN_df.drop(columns=['Code'], inplace=True)

print(DBSCAN_df.head())

# use original data (pre pre-processing)
DBSCAN_df_original = pd.DataFrame(df_clean, columns=["recency", "frequency", "monetary", "State"])
DBSCAN_df_original['Cluster'] = dbscan_labels
DBSCAN_df_original['State'] = DBSCAN_df_original['State'].map(state_mapping)
print(DBSCAN_df_original.head())

RFMD_final.rename(columns={'DBSCAN_Cluster': 'Cluster'}, inplace=True)
RFMD_final.drop(columns=['Code'], inplace=True)

utils.plot_3d_clusters(RFMD_final, "DBSCAN")

# summary
print("DBSCAN cluster summary:")
print(utils.summarize_cluster(RFMD_final, True))

print("DBSCAN cluster summary (Original Data):")
print(utils.summarize_cluster(DBSCAN_df_original))

utils.summarize_cluster_v2(DBSCAN_df_original)

utils.export_pickle(RFMD_final, "rfmd_dbscan.pkl")
utils.export_pickle(DBSCAN_df_original, "rfmd_dbscan_clean.pkl")