import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import utils
from logger import Logger
from benchmark import Benchmark
import plotly.graph_objects as go
import numpy as np
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

utils.widen_output(pd)
RFMD_final = utils.import_pickle('../08-rfmd-final-processing/rfmd_final.pkl')
df_clean = utils.import_pickle('../05-outlier/rfmd_clean.pkl')
state_mapping = utils.import_pickle('../08-rfmd-final-processing/state_mapping.pkl')
# manhattan_df = utils.import_pickle('dbscan_manhattan_results.pkl')
# euclidean_df = utils.import_pickle('dbscan_euclidean_results.pkl')

logger = Logger()
benchmark = Benchmark(logger)

logger.print("DBSCAN Hyperparameter Optimization for RFM Data")
logger.print("=" * 50)

# Prepare data
RFMD_final = RFMD_final.drop(columns=['Code', 'State'])
numeric_df = RFMD_final.select_dtypes(include=['float64', 'int64'])
logger.print(f"Dataset shape: {numeric_df.shape}")

def find_optimal_eps(data, metric='euclidean', k=5, plot_path=None):
    # Compute k-distances
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1)  # k+1 because we exclude self
    neighbors.fit(data)
    distances, _ = neighbors.kneighbors(data)
    kdistances = np.sort(distances[:, k])  # Take the k-th distance (0-indexed, so k gives us k+1-th neighbor)

    # KneeLocator on the whole data
    x = range(len(kdistances))
    knee_locator = KneeLocator(
        x,
        kdistances,
        curve='convex',
        direction='increasing',
        S=3,
    )
    knee_x = knee_locator.elbow

    # Add null check and handle potential None
    if knee_x is not None:
        optimal_eps = kdistances[knee_x]
        logger.print(f"  k={k}: Optimal eps found at index {knee_x}: {optimal_eps:.4f}")
    else:
        optimal_eps = None
        logger.print(f"  k={k}: No elbow found in the k-distance curve")

    # Plotting (optional)
    if plot_path:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(x),
            y=kdistances,
            mode='lines',
            name='K-distance Curve',
            line=dict(color='blue', width=2),
            hovertemplate='Point: %{x}<br>Distance: %{y:.4f}<extra></extra>'
        ))

        # Knee point
        if knee_x is not None:
            fig.add_trace(go.Scatter(
                x=[knee_x],
                y=[optimal_eps],
                mode='markers+text',
                name='Elbow Point',
                marker=dict(color='red', size=12),
                text=[f'Elbow ({knee_x}, {optimal_eps:.4f})'],
                textposition='middle left',
                textfont=dict(size=20),
                hovertemplate='Elbow Point<br>Index: %{x}<br>Eps: %{y:.4f}<extra></extra>'
            ))

        fig.update_layout(
            title=dict(
                text=f"K-distance Graph - {metric.title()} (k={k})",
                font=dict(size=20)
            ),
            xaxis=dict(
                title=dict(
                    text="Points sorted by distance",
                    font=dict(size=16)
                ),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title=dict(
                    text=f"{metric.title()} distance to {k}th nearest neighbor",
                    font=dict(size=16)
                ),
                tickfont=dict(size=14)
            ),
            legend=dict(
                font=dict(size=14)
            ),
            font=dict(size=14),
            hovermode='closest',
            template='plotly_white'
        )
        fig.write_html(plot_path)

    return optimal_eps, kdistances


# 2. Compute eps values for different metrics and k values
metrics = ['euclidean', 'manhattan']
k_range = range(12, 13)  # k values from 6 to 100

logger.print("\nFinding optimal eps values for different k values:")

# 3. Create parameter combinations
configurations = []
best_config = None
best_score = float('inf')
results = []

for metric in metrics:
    logger.print(f"\nProcessing metric: {metric}")

    for k in k_range:
        # Find optimal eps for this specific k
        eps, kdistances = find_optimal_eps(
            numeric_df.values,
            metric=metric,
            k=k,
            plot_path=f'kdistance_graph_{metric}_k{k}.html'
        )

        # Skip if no eps found
        if eps is None:
            logger.print(f"  Skipping k={k} due to no optimal eps found")
            continue

        # Use this k as min_samples and the corresponding eps
        logger.print(" " + "=" * 50)
        logger.print(f"  Testing: {metric} eps={eps:.6f} min_samples={k}")

        try:
            db = DBSCAN(eps=eps, min_samples=k, metric=metric, n_jobs=-1)
            labels = db.fit_predict(numeric_df.values)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(labels)

            # Calculate clustering metrics (only if we have valid clusters)
            calinski_score = np.nan
            davies_bouldin = np.nan
            silhouette_avg = np.nan
            composite_score = np.inf

            if n_clusters >= 2:
                # Filter out noise points for metric calculation
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 0:
                    data_clean = numeric_df.values[non_noise_mask]
                    labels_clean = labels[non_noise_mask]

                    # Only calculate if we have more than 1 cluster after removing noise
                    if len(set(labels_clean)) >= 2:
                        try:
                            calinski_score = calinski_harabasz_score(data_clean, labels_clean)
                            davies_bouldin = davies_bouldin_score(data_clean, labels_clean)
                            silhouette_avg = silhouette_score(data_clean, labels_clean)

                            # Composite score (lower is better)
                            # Normalize scores: Calinski (higher=better), Davies-Bouldin (lower=better), Silhouette (higher=better)
                            composite_score = (
                                    (1 / (calinski_score + 1e-8)) +  # Invert Calinski (higher is better)
                                    davies_bouldin +  # Davies-Bouldin (lower is better)
                                    (1 / (silhouette_avg + 1e-8)) +  # Invert Silhouette (higher is better)
                                    (noise_ratio * 2)  # Penalize noise
                            )
                        except Exception as metric_error:
                            logger.print(f"    → Metric calculation error: {str(metric_error)}")

            result = {
                'metric': metric,
                'eps': eps,
                'min_samples': k,
                'n_clusters': n_clusters,
                'noise_ratio': noise_ratio,
                'n_noise': n_noise,
                'calinski_score': calinski_score,
                'davies_bouldin': davies_bouldin,
                'silhouette_score': silhouette_avg,
                'composite_score': composite_score,
                'labels': labels
            }
            results.append(result)

            logger.print(f"    → clusters={n_clusters:3d} noise={noise_ratio:5.1%} "
                         f"CH={calinski_score:.2f} DB={davies_bouldin:.3f} Sil={silhouette_avg:.3f}")

            # Evaluate this configuration using composite score
            if 2 <= n_clusters <= 25 and not np.isinf(composite_score):
                if composite_score < best_score:
                    best_score = composite_score
                    best_config = result.copy()
                    logger.print(f"    → NEW BEST! Composite Score: {composite_score:.4f}")

        except Exception as e:
            logger.print(f"    → ERROR: {str(e)}")

# 4. Show best result
logger.print("\n" + "=" * 50)
logger.print("OPTIMAL HYPERPARAMETERS")
logger.print("=" * 50)

if best_config:
    logger.print(f"✅ Best Configuration Found:")
    logger.print(f"   Metric: {best_config['metric']}")
    logger.print(f"   eps: {best_config['eps']:.6f}")
    logger.print(f"   min_samples: {best_config['min_samples']}")
    logger.print(f"   Composite Score: {best_score:.6f}")
    logger.print(f"   ")
    logger.print(f"   Clustering Results:")
    logger.print(f"   - Number of clusters: {best_config['n_clusters']}")
    logger.print(f"   - Noise ratio: {best_config['noise_ratio']:.4f}")
    logger.print(f"   - Number of noise points: {best_config['n_noise']}")
    logger.print(f"   ")
    logger.print(f"   Quality Metrics:")
    logger.print(f"   - Calinski-Harabasz Score: {best_config['calinski_score']:.4f}")
    logger.print(f"   - Davies-Bouldin Score: {best_config['davies_bouldin']:.4f}")
    logger.print(f"   - Silhouette Score: {best_config['silhouette_score']:.4f}")
else:
    logger.print("❌ No valid configuration found.")
    logger.print("Please check the logs for more details.")

# 5. Optional: Show top 10 configurations
logger.print("\n" + "=" * 50)
logger.print("TOP 10 CONFIGURATIONS")
logger.print("=" * 50)

valid_results = [r for r in results if 2 <= r['n_clusters'] <= 25]
valid_results.sort(key=lambda x: x['noise_ratio'] + (0.1 if x['n_clusters'] < 4 or x['n_clusters'] > 15 else 0))

for i, result in enumerate(valid_results[:10]):
    score = result['noise_ratio'] + (0.1 if result['n_clusters'] < 4 or result['n_clusters'] > 15 else 0)
    logger.print(f"{i + 1:2d}. {result['metric']:9s} eps={result['eps']:8.4f} "
                 f"min_samples={result['min_samples']:3d} clusters={result['n_clusters']:3d} "
                 f"noise={result['noise_ratio']:5.1%} score={score:.4f}")

# 6. Plot evaluation metrics - Option 1: Adapt for existing function

def convert_results_for_plotting(results):
    """Convert DBSCAN results to format expected by plot_evaluation_metrics function"""
    if results is None or len(results) == 0:
        logger.print("No results to plot")
        return None, None

    # Create DataFrame from results
    df = pd.DataFrame(results)

    # Create params column with 'k' key (using min_samples as k for x-axis)
    df['params'] = df.apply(lambda row: {
        'k': row['min_samples'],  # This will be used as x-axis
        'eps': row['eps'],
        'min_samples': row['min_samples'],
        'metric': row['metric']
    }, axis=1)

    # Rename columns to match expected names in plot function
    column_mapping = {
        'silhouette_score': 'silhouette',
        'davies_bouldin': 'davies_bouldin',
        'calinski_score': 'calinski_harabasz'
    }
    df.rename(columns=column_mapping, inplace=True)

    # Filter out rows with NaN values in key metrics
    df = df.dropna(subset=['silhouette', 'davies_bouldin', 'calinski_harabasz'])

    # Normalize metric column values
    df['metric'] = df['metric'].str.lower()

    # Split by metric
    manhattan_df = df[df['metric'] == 'manhattan'].reset_index(drop=True)
    euclidean_df = df[df['metric'] == 'euclidean'].reset_index(drop=True)

    return manhattan_df, euclidean_df

# Convert results for plotting
manhattan_df, euclidean_df = convert_results_for_plotting(results)

logger.print("\nConverted results for plotting:")

if euclidean_df is not None and not euclidean_df.empty:
    logger.print(f"  Euclidean DataFrame Head:\n{euclidean_df.head()}")
else:
    logger.print("  Euclidean DataFrame is empty or None.")

if manhattan_df is not None and not manhattan_df.empty:
    logger.print(f"  Manhattan DataFrame Head:\n{manhattan_df.head()}")
else:
    logger.print("  Manhattan DataFrame is empty or None.")

# utils.export_pickle(manhattan_df, 'dbscan_manhattan_results.pkl')
# utils.export_pickle(euclidean_df, 'dbscan_euclidean_results.pkl')

# Plot Euclidean results
logger.print("\nPlotting Euclidean results")
utils.plot_evaluation_metrics(euclidean_df, 'euclidean_dbscan')
#
# # Plot Manhattan results
logger.print("\nPlotting Manhattan results")
utils.plot_evaluation_metrics(manhattan_df, 'manhattan_dbscan')

# Save best configuration
# utils.export_pickle(results, 'dbscan_results.pkl')