import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
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

# Initialize logger and benchmark
logger = Logger()
benchmark = Benchmark(logger)

logger.print("Starting domain knowledge-informed DBSCAN clustering analysis...")

# Assume numeric_df is already loaded and pre-processed
# If working with a new dataset:
# numeric_df = RFMD_final.drop(columns=['State', 'Code'])  # Assuming these are non-numeric columns

# 1. Exploratory Analysis for Domain Knowledge Application
logger.print("\n1. Exploring RFM data distributions for domain knowledge application")
logger.print("-----------------------------------------------------------------")

# Basic statistics
logger.print("\nBasic statistics of RFM features:")
stats = numeric_df.describe()
logger.print(stats)

# Visualize distributions
plt.figure(figsize=(15, 5))
for i, col in enumerate(numeric_df.columns):
    plt.subplot(1, 3, i + 1)
    sns.histplot(numeric_df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('rfm_distributions.png')

# 2. Distance Metrics Selection
logger.print("\n2. Distance Metrics Selection")
logger.print("----------------------------")
logger.print("Available distance metrics for DBSCAN:")
logger.print("1. Euclidean (default): Standard distance, equal weight to all dimensions")
logger.print("2. Manhattan: Less sensitive to outliers, sum of absolute differences")
logger.print("3. Cosine: Measures angle between vectors, captures directional similarity")
logger.print("4. Mahalanobis: Accounts for correlations between features")

# 3. Finding optimal eps for each distance metric
logger.print("\n3. Finding optimal eps values for different distance metrics")
logger.print("----------------------------------------------------------")

def compute_pairwise_distances_in_chunks(data, metric, chunk_size=1000, logger=None):
    n = data.shape[0]
    distances = lil_matrix((n, n))  # Use a sparse matrix to store distances
    total_chunks = (n // chunk_size + (1 if n % chunk_size != 0 else 0)) ** 2
    chunk_counter = 0

    if logger:
        logger.print(f"Starting pairwise distance computation in chunks (total chunks: {total_chunks})")

    for i in range(0, n, chunk_size):
        for j in range(0, n, chunk_size):
            if logger:
                logger.print(f"Processing chunk ({chunk_counter + 1}/{total_chunks}): rows {i}-{min(i + chunk_size, n)}, "
                             f"columns {j}-{min(j + chunk_size, n)}")
            chunk_distances = pairwise_distances(
                data[i:i+chunk_size], data[j:j+chunk_size], metric=metric
            )
            distances[i:i+chunk_size, j:j+chunk_size] = chunk_distances
            chunk_counter += 1

    if logger:
        logger.print("Pairwise distance computation completed.")
    return distances.tocsr()  # Convert to CSR format for efficient operations

def find_optimal_eps(data, metric='euclidean', n_neighbors=5, quantile=0.95, chunk_size=1000, logger=None):
    """
    Find optimal eps value using the k-distance graph with chunked computation.

    Parameters:
    - data: feature matrix
    - metric: distance metric ('euclidean', 'manhattan', 'cosine', etc.)
    - n_neighbors: number of neighbors to consider
    - quantile: quantile to use for finding elbow point
    - chunk_size: size of chunks for pairwise distance computation
    - logger: logger instance for progress tracking

    Returns:
    - optimal_eps: suggested eps value
    - kdistances: sorted k-distances for plotting
    """
    if logger:
        logger.print(f"Finding optimal eps using metric: {metric}")

    if metric == 'mahalanobis':
        V = np.cov(data.T)
        VI = np.linalg.inv(V)

        def mahalanobis_distance(x, y):
            diff = x - y
            return np.sqrt(np.dot(np.dot(diff, VI), diff.T))

        distances_matrix = compute_pairwise_distances_in_chunks(data, metric=mahalanobis_distance, chunk_size=chunk_size, logger=logger)
        sorted_distances = np.sort(distances_matrix, axis=1)
        kdistances = sorted_distances[:, n_neighbors]
    else:
        distances_matrix = compute_pairwise_distances_in_chunks(data, metric=metric, chunk_size=chunk_size, logger=logger)
        sorted_distances = np.sort(distances_matrix, axis=1)
        kdistances = sorted_distances[:, n_neighbors]

    kdistances = np.sort(kdistances)
    optimal_eps = kdistances[int(len(kdistances) * quantile)]

    if logger:
        logger.print(f"Optimal eps value found: {optimal_eps:.4f}")
    return optimal_eps, kdistances

# List of metrics to try
metrics = ['euclidean', 'manhattan', 'cosine', 'mahalanobis']

# Find optimal eps for each metric
eps_results = {}
logger.print("\nOptimal eps values for different metrics:")

# Find eps for standard metrics
for metric in metrics:
    if metric == 'mahalanobis':
        optimal_eps, kdistances = find_optimal_eps(numeric_df.values, metric=metric)
    else:
        optimal_eps, kdistances = find_optimal_eps(numeric_df.values, metric=metric)

    eps_results[metric] = optimal_eps
    logger.print(f"  * {metric}: {optimal_eps:.4f}")

    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(kdistances)), kdistances, 'b-')
    plt.axhline(y=optimal_eps, color='r', linestyle='--')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{metric} distance to {5}th nearest neighbor')
    plt.title(f'K-distance graph ({metric} metric)')
    plt.grid(True)
    plt.savefig(f'kdistance_graph_{metric}.png')

# 4. Determine minPts (min_samples) based on domain knowledge
logger.print("\n4. Determining minPts (min_samples) based on domain knowledge")
logger.print("-----------------------------------------------------------")
logger.print("Considerations for setting minPts:")
logger.print("  * Dataset size (total points):", len(numeric_df))
logger.print("  * Rule of thumb: minPts ≥ dim + 1:", numeric_df.shape[1] + 1)
logger.print("  * For dense regions, consider higher minPts (better clusters)")
logger.print("  * For sparse regions, consider lower minPts (more sensitivity)")
logger.print("  * Minimum meaningful business segment size")

# Suggest some values based on dataset characteristics
pct_1 = int(len(numeric_df) * 0.001)  # 0.1% of data points
pct_01 = int(len(numeric_df) * 0.0001)  # 0.01% of data points
dim_rule = numeric_df.shape[1] + 1  # dimensionality + 1

logger.print("\nSuggested minPts values based on dataset characteristics:")
logger.print(f"  * Minimum (dims + 1): {dim_rule}")
logger.print(f"  * 0.01% of dataset: {pct_01}")
logger.print(f"  * 0.1% of dataset: {pct_1}")
logger.print(f"  * Conservative default: {max(dim_rule, pct_01)}")

# 5. Run DBSCAN with different combinations
logger.print("\n5. Running DBSCAN with different configurations based on domain knowledge")
logger.print("-----------------------------------------------------------------------")

# Combinations to try
configurations = [
    # Standard metrics with their optimal eps values
    ('euclidean', eps_results['euclidean'], max(dim_rule, pct_01)),
    ('manhattan', eps_results['manhattan'], max(dim_rule, pct_01)),
    ('cosine', eps_results['cosine'], max(dim_rule, pct_01)),
    ('mahalanobis', eps_results['mahalanobis'], max(dim_rule, pct_01)),

    # Try different minPts values with Euclidean distance
    ('euclidean', eps_results['euclidean'], dim_rule),  # Minimum theoretical value
    ('euclidean', eps_results['euclidean'], pct_1),  # Larger segment size

    # Try different eps values with Euclidean distance
    ('euclidean', eps_results['euclidean'] * 0.8, max(dim_rule, pct_01)),  # Smaller neighborhoods
    ('euclidean', eps_results['euclidean'] * 1.2, max(dim_rule, pct_01))  # Larger neighborhoods
]


# Create helper function for Mahalanobis distance
def get_precomputed_distances(data, metric_name):
    """Generate a precomputed distance matrix for custom metrics"""
    if metric_name == 'mahalanobis':
        V = np.cov(data.T)
        VI = np.linalg.inv(V)

        def mahalanobis_distance(x, y):
            diff = x - y
            return np.sqrt(np.dot(np.dot(diff, VI), diff.T))

        return pairwise_distances(data, metric=mahalanobis_distance)
    else:
        return None


# Run DBSCAN with different configurations
results = []
for metric_name, eps, min_samples in configurations:
    logger.print(f"\nRunning DBSCAN with {metric_name}, eps={eps:.4f}, min_samples={min_samples}")
    benchmark.start_benchmark()

    # For standard metrics, we can use them directly
    if metric_name in ['euclidean', 'manhattan', 'cosine']:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric_name, n_jobs=-1).fit(numeric_df.values)

    # For custom metrics, we need to precompute the distance matrix
    else:
        dist_matrix = get_precomputed_distances(numeric_df.values, metric_name)
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1).fit(dist_matrix)

    benchmark.end_benchmark()

    # Analyze results
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_ratio = n_noise / len(labels)

    logger.print(f"  * Number of clusters: {n_clusters}")
    logger.print(f"  * Number of noise points: {n_noise} ({noise_ratio:.2%} of data)")

    # Record the results
    results.append({
        'metric': metric_name,
        'eps': eps,
        'min_samples': min_samples,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': noise_ratio,
        'labels': labels
    })

    # If we have a good clustering result (multiple clusters, reasonable noise),
    # analyze the clusters in more detail
    if n_clusters >= 2 and noise_ratio < 0.1:
        # Add cluster labels to original dataframe
        temp_df = numeric_df.copy()
        temp_df['Cluster'] = labels

        # Analyze cluster characteristics
        logger.print("\nCluster Characteristics (Mean Values):")
        cluster_summary = temp_df.groupby('Cluster').mean()
        logger.print(cluster_summary)

        # Cluster sizes
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        logger.print("\nCluster Sizes:")
        for cluster_id, size in cluster_sizes.items():
            if cluster_id == -1:
                logger.print(f"Noise points: {size} samples ({size / len(labels):.2%})")
            else:
                logger.print(f"Cluster {cluster_id}: {size} samples ({size / len(labels):.2%})")

        # Visualize the clusters (using PCA for dimensionality reduction)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numeric_df.values)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'DBSCAN Clusters ({metric_name}, eps={eps:.4f}, min_samples={min_samples})')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.savefig(f'dbscan_clusters_{metric_name}.png')

# 6. Compare and recommend the best configuration
logger.print("\n6. Comparing DBSCAN configurations and making recommendations")
logger.print("----------------------------------------------------------")

# Convert results to DataFrame for easier comparison
results_df = pd.DataFrame([
    {
        'Metric': r['metric'],
        'Epsilon': r['eps'],
        'Min_Samples': r['min_samples'],
        'Num_Clusters': r['n_clusters'],
        'Noise_Points': r['n_noise'],
        'Noise_Percentage': r['noise_ratio'] * 100
    } for r in results
])

logger.print("\nComparison of DBSCAN configurations:")
logger.print(results_df)

# Find the best configuration (subjective - could be different based on business needs)
# Here, we define "best" as having multiple clusters and low noise
valid_results = results_df[results_df['Num_Clusters'] >= 2]
if not valid_results.empty:
    # Sort by noise percentage (ascending)
    valid_results = valid_results.sort_values('Noise_Percentage')
    best_config = valid_results.iloc[0]

    logger.print("\nRecommended configuration based on cluster quality:")
    logger.print(f"  * Metric: {best_config['Metric']}")
    logger.print(f"  * Epsilon: {best_config['Epsilon']:.4f}")
    logger.print(f"  * Min_Samples: {int(best_config['Min_Samples'])}")
    logger.print(f"  * Number of clusters: {int(best_config['Num_Clusters'])}")
    logger.print(f"  * Noise percentage: {best_config['Noise_Percentage']:.2f}%")

    # Use the best configuration for final clustering
    best_idx = results_df[
        (results_df['Metric'] == best_config['Metric']) &
        (results_df['Epsilon'] == best_config['Epsilon']) &
        (results_df['Min_Samples'] == best_config['Min_Samples'])
        ].index[0]

    best_labels = results[best_idx]['labels']

    # Create final output with cluster labels
    final_df = numeric_df.copy()
    final_df['Cluster'] = best_labels

    # Save the clustered data
    utils.export_pickle(final_df, 'rfm_clustered.pkl')

    logger.print("\nFinal clustering saved to 'rfm_clustered.pkl'")
else:
    logger.print("\nNo configuration produced multiple clusters with acceptable noise levels.")
    logger.print("Consider adjusting the parameter ranges or distance metrics.")

logger.print("\nDBSCAN clustering analysis with domain knowledge completed.")