import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import utils
from logger import Logger
from benchmark import Benchmark

utils.widen_output(pd)
RFMD_final = utils.import_pickle('../08-rfmd-final-processing/rfmd_final.pkl')
df_clean = utils.import_pickle('../05-outlier/rfmd_clean.pkl')
state_mapping = utils.import_pickle('../08-rfmd-final-processing/state_mapping.pkl')
logger = Logger()
benchmark = Benchmark(logger)
# Use the already loaded data from your code
logger.print("Starting DBSCAN clustering analysis...")

# drop code and state columns
RFMD_final = RFMD_final.drop(columns=['Code', 'State'])

numeric_df = RFMD_final.select_dtypes(include=['float64', 'int64'])

# Since data is already pre-processed, we'll use it directly
logger.print("Using pre-processed data directly...")
features = numeric_df.values


# Step 2: Find optimal DBSCAN parameters
# For large datasets (100k), we need to be careful with eps and min_samples values
# Let's try a few different parameter combinations

# Function to evaluate DBSCAN results
def evaluate_dbscan(eps, min_samples):
    benchmark.start_benchmark()
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(features)
    benchmark.end_benchmark()

    # Number of clusters (excluding noise points with label -1)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = list(db.labels_).count(-1)

    logger.print(f"DBSCAN with eps={eps}, min_samples={min_samples}:")
    logger.print(f"  * Number of clusters: {n_clusters}")
    logger.print(f"  * Number of noise points: {n_noise} ({n_noise / len(db.labels_) * 100:.2f}% of data)")

    return db, n_clusters, n_noise


# Try a few parameter combinations
parameters = [
    (0.5, 50),  # More restrictive
    (0.75, 30),  # Medium
    (1.0, 20)  # More inclusive
]

results = []
for eps, min_samples in parameters:
    db, n_clusters, n_noise = evaluate_dbscan(eps, min_samples)
    results.append((eps, min_samples, db, n_clusters, n_noise))

# Select the best parameters (this is subjective - we want a reasonable number of clusters with not too many noise points)
# For this example, let's choose the one with the highest number of clusters but less than 30% noise
valid_results = [(eps, min_samples, db, n_clusters, n_noise)
                 for eps, min_samples, db, n_clusters, n_noise in results
                 if n_noise / len(db.labels_) < 0.3]

if valid_results:
    # Sort by number of clusters (descending)
    valid_results.sort(key=lambda x: x[3], reverse=True)
    best_eps, best_min_samples, best_db, n_clusters, n_noise = valid_results[0]
    logger.print(f"Selected parameters: eps={best_eps}, min_samples={best_min_samples}")
else:
    # If all have high noise, select the one with the lowest noise
    results.sort(key=lambda x: x[4])
    best_eps, best_min_samples, best_db, n_clusters, n_noise = results[0]
    logger.print(
        f"All parameter combinations resulted in high noise. Selected: eps={best_eps}, min_samples={best_min_samples}")

# Step 3: Apply DBSCAN with the best parameters
logger.print(f"Running final DBSCAN with eps={best_eps}, min_samples={best_min_samples}...")
benchmark.start_benchmark()
best_db = DBSCAN(eps=best_eps, min_samples=best_min_samples, n_jobs=-1).fit(features)
benchmark.end_benchmark()

# Add cluster labels to the original dataframe
RFMD_final['Cluster'] = best_db.labels_

# Step 4: Analyze the clusters
logger.print("\nCluster Analysis:")
cluster_counts = RFMD_final['Cluster'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    if cluster_id == -1:
        logger.print(f"Noise points: {count} samples ({count / len(RFMD_final) * 100:.2f}%)")
    else:
        logger.print(f"Cluster {cluster_id}: {count} samples ({count / len(RFMD_final) * 100:.2f}%)")

# Step 5: Summarize the clusters
logger.print("\nCluster Characteristics (Mean Values):")
cluster_summary = RFMD_final.groupby('Cluster').mean()
logger.print(cluster_summary)

# Step 6: Visualize the clusters using PCA to reduce to 2D
logger.print("\nVisualizing clusters using PCA...")
benchmark.start_benchmark()
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)

# Create a DataFrame for easier plotting
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['Cluster'] = best_db.labels_

# Plot the clusters
plt.figure(figsize=(10, 8))
plt.scatter(data=pca_df, x='PC1', y='PC2', c='Cluster', cmap='viridis', s=50, alpha=0.5)
plt.title(f'DBSCAN Clusters (eps={best_eps}, min_samples={best_min_samples})')
plt.savefig('dbscan_clusters.png')

# Optional: If State information is important, analyze the state distribution in each cluster
if 'State' in RFMD_final.columns:
    logger.print("\nTop 3 States in each cluster:")
    for cluster_id in sorted(set(RFMD_final['Cluster'])):
        if cluster_id == -1:
            continue  # Skip noise points
        cluster_data = RFMD_final[RFMD_final['Cluster'] == cluster_id]
        state_distribution = cluster_data['State'].value_counts().nlargest(3)
        logger.print(f"Cluster {cluster_id}:")
        for state, count in state_distribution.items():
            percentage = count / len(cluster_data) * 100
            logger.print(f"  * {state}: {count} samples ({percentage:.2f}%)")

# Step 7: Analyze RFM characteristics of each cluster
logger.print("\nRFM Analysis by Cluster:")
rfm_columns = [col for col in RFMD_final.columns if col not in ['Cluster', 'State', 'Code']]

for cluster_id in sorted(set(RFMD_final['Cluster'])):
    if cluster_id == -1:
        logger.print(f"Noise Points RFM Profile:")
    else:
        logger.print(f"Cluster {cluster_id} RFM Profile:")

    cluster_data = RFMD_final[RFMD_final['Cluster'] == cluster_id]
    for column in rfm_columns:
        mean_val = cluster_data[column].mean()
        logger.print(f"  * {column}: {mean_val:.2f}")

benchmark.end_benchmark()
logger.print("DBSCAN clustering analysis completed.")