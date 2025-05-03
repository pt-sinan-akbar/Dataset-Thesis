import pandas
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pickle

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