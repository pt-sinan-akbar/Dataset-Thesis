import pandas
import pandas as pd
from kmodes.kprototypes import KPrototypes
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture

with open('../08-rfmd-final-processing/encoded_to_state.pkl', 'rb') as file:
    encoded_to_state = pickle.load(file)

def plot_3d_clusters(df: pd.DataFrame, algorithm_name: str = "Clustering", show_plot: bool = False):
    if not all(col in df.columns for col in ['recency', 'frequency', 'monetary', 'Cluster']):
        raise ValueError("DataFrame must contain 'recency', 'frequency', 'monetary', and 'Cluster' columns.")

    num_clusters = df['Cluster'].nunique()

    # Select an appropriate colormap
    if num_clusters <= 10:
        cmap = ListedColormap(plt.colormaps['tab10'].colors[:num_clusters])
    elif num_clusters <= 20:
        cmap = ListedColormap(plt.colormaps['tab20'].colors[:num_clusters])
    else:
        raise ValueError("Too many clusters for a discrete colormap. Consider using a different visual strategy.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        df['recency'],
        df['frequency'],
        df['monetary'],
        c=df['Cluster'],
        cmap=cmap,
        s=20,
        alpha=0.7
    )
    ax.set_xlabel('recency')
    ax.set_ylabel('frequency')
    ax.set_zlabel('monetary')
    plt.title(f'3D Cluster Visualization ({algorithm_name})')

    # Add custom legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label=f'Cluster {i}',
                   markerfacecolor=cmap(i), markersize=8)
        for i in range(num_clusters)
    ]
    ax.legend(handles=handles, title="Clusters", loc='upper left')

    plt.tight_layout()
    if show_plot:
        plt.show()
    plt.savefig(algorithm_name + '_3d_cluster.png', dpi=300, bbox_inches='tight')

# Function to summarize cluster customer percentages and state distribution, and per-cluster statistic 
# p.s. for data with numerical state pls give remap_state=True
def summarize_cluster_v2(df, remap_state=False):
    print("\nSummarizing cluster information v2...")
    
    print("\nCluster percentages and state distribution")
    cluster_summary = df.groupby('Cluster').size().reset_index(name='Number of Customer')
    total_customers = cluster_summary['Number of Customer'].sum()
    cluster_summary['Percentage (%)'] = (cluster_summary['Number of Customer'] / total_customers * 100).round(2)
    
    state_summary = df.groupby(['Cluster', 'State']).size().unstack(fill_value=0)
    if remap_state:
        state_summary.columns = state_summary.columns.map(encoded_to_state)
    
    combined_summary = pd.concat([cluster_summary.set_index('Cluster'), state_summary], axis=1)
    print(combined_summary)

    print("\nPer-cluster statistic")
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        stats = cluster_data.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T
        stats = stats[stats.index != 'Cluster']
        print('\nCluster', cluster)
        print(stats)

# cluster summary
def summarize_cluster(df, remap_state=True, debug=False):
    cluster_size = df['Cluster'].value_counts().sort_index()
    total_samples = len(df)
    percentages = ((cluster_size / total_samples) * 100).round(2)

    summary = df.groupby('Cluster').agg({
        **{col: ['mean'] for col in df.columns if col != 'Cluster' and col != 'State'}
    })
    if debug:
        print("satu")
        print(summary.head())

    # Add count and percentage columns
    summary['Number of Customer'] = cluster_size
    summary['Percentage (%)'] = percentages
    if debug:
        print("dua")
        print(summary.head())

    # print state
    state_summary = df.groupby('Cluster')['State'].value_counts().unstack().fillna(0)
    # use state_mapping to decode the state
    if remap_state:
        state_summary.columns = state_summary.columns.map(encoded_to_state)
    if debug:
        print("tiga")
        print(state_summary.head())

    summary = pd.concat([summary, state_summary], axis=1)
    if debug:
        print("empat")
        print(summary.head())

    # rename columns
    summary.columns = ['Mean R', 'Mean F', 'Mean M', 'Number of Customer', 'Percentage (%)'] + list(state_summary.columns)
    if debug:
        print("lima")
        print(summary.head())

    return summary

def widen_output(pd_loc: pandas):
    pd_loc.set_option('display.max_columns', None)
    pd_loc.set_option('display.max_colwidth', None)
    return pd_loc
    
def import_pickle(pickle_location):
    print("Importing " + pickle_location)
    with open(pickle_location, 'rb') as dataset_file:
        dataset = pickle.load(dataset_file)
        return dataset
    
def export_pickle(data, pickle_name, is_not_df: bool = False):
    print("Saving to", pickle_name)
    if is_not_df:
        pickle.dump(data, open(pickle_name, 'wb'))
    else:
        data.to_pickle(pickle_name)


def evaluation_metrics(df, algorithm, cluster_range=None, eps_values=None, min_samples=5):
    results = {
        'algorithm': [],
        'params': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }

    X_numeric = pd.DataFrame(df, columns=['recency', 'frequency', 'monetary'])
    X_categorical = [3]

    if algorithm == "KMeans":
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, n_init=10)
            labels = kmeans.fit_predict(X_numeric)

            results['algorithm'].append('KMeans')
            results['params'].append({'k': k})
            results['silhouette'].append(silhouette_score(X_numeric, labels))
            results['davies_bouldin'].append(davies_bouldin_score(X_numeric, labels))
            results['calinski_harabasz'].append(calinski_harabasz_score(X_numeric, labels))

    elif algorithm == "DBSCAN":
        if eps_values is None:
            raise ValueError("eps_values must be provided for DBSCAN.")
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_numeric)

            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if num_clusters > 1:
                core_mask = labels != -1
                silhouette = silhouette_score(X_numeric[core_mask], labels[core_mask])
                db_score = davies_bouldin_score(X_numeric[core_mask], labels[core_mask])
                ch_score = calinski_harabasz_score(X_numeric[core_mask], labels[core_mask])
            else:
                silhouette, db_score, ch_score = np.nan, np.nan, np.nan

            results['algorithm'].append('DBSCAN')
            results['params'].append({'eps': eps, 'min_samples': min_samples})
            results['silhouette'].append(silhouette)
            results['davies_bouldin'].append(db_score)
            results['calinski_harabasz'].append(ch_score)

    elif algorithm == "GMM":
        for k in cluster_range:
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
            labels = gmm.fit_predict(X_numeric)

            results['algorithm'].append('GMM')
            results['params'].append({'n_components': k})
            results['silhouette'].append(silhouette_score(X_numeric, labels))
            results['davies_bouldin'].append(davies_bouldin_score(X_numeric, labels))
            results['calinski_harabasz'].append(calinski_harabasz_score(X_numeric, labels))

    elif algorithm == "Hierarchical":
        for k in cluster_range:
            hierarchical = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
            labels = hierarchical.fit_predict(X_numeric)

            results['algorithm'].append('Hierarchical')
            results['params'].append({'n_clusters': k})
            results['silhouette'].append(silhouette_score(X_numeric, labels))
            results['davies_bouldin'].append(davies_bouldin_score(X_numeric, labels))
            results['calinski_harabasz'].append(calinski_harabasz_score(X_numeric, labels))

    elif algorithm == "KPrototypes":
        for k in cluster_range:
            kproto = KPrototypes(n_clusters=k, init='Huang', gamma=1.0, n_init=5, verbose=0)
            labels = kproto.fit_predict(df, categorical=X_categorical)

            results['algorithm'].append('KPrototypes')
            results['params'].append({'k': k})
            results['silhouette'].append(silhouette_score(X_numeric, labels))
            results['davies_bouldin'].append(davies_bouldin_score(X_numeric, labels))
            results['calinski_harabasz'].append(calinski_harabasz_score(X_numeric, labels))

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return pd.DataFrame(results)


def plot_evaluation_metrics(results):
    # Determine whether to use 'k' or 'eps' based on the algorithm
    if 'k' in results['params'][0]:
        results['x_param'] = results['params'].apply(lambda x: x.get('k') or x.get('n_components'))
        x_label = 'Number of Clusters (k)'
    elif 'eps' in results['params'][0]:
        results['x_param'] = results['params'].apply(lambda x: x.get('eps'))
        x_label = 'Epsilon (eps)'
    else:
        raise ValueError("Unsupported parameter in 'params'. Expected 'k' or 'eps'.")

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot Silhouette Score (higher is better)
    axs[0].plot(results['x_param'], results['silhouette'], marker='o', linestyle='-', color='blue')
    axs[0].set_title('Silhouette Score (Higher is Better)')
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel('Silhouette Score')
    axs[0].grid(True)

    # Plot Davies-Bouldin Score (lower is better)
    axs[1].plot(results['x_param'], results['davies_bouldin'], marker='o', linestyle='-', color='red')
    axs[1].set_title('Davies-Bouldin Score (Lower is Better)')
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel('Davies-Bouldin Score')
    axs[1].grid(True)

    # Plot Calinski-Harabasz Score (higher is better)
    axs[2].plot(results['x_param'], results['calinski_harabasz'], marker='o', linestyle='-', color='green')
    axs[2].set_title('Calinski-Harabasz Score (Higher is Better)')
    axs[2].set_xlabel(x_label)
    axs[2].set_ylabel('Calinski-Harabasz Score')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')