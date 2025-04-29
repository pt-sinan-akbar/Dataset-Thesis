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

def summarize_cluster2(df, remap_state=True, debug=False):
    import numpy as np

    cluster_size = df['Cluster'].value_counts().sort_index()
    total_samples = len(df)
    percentages = ((cluster_size / total_samples) * 100).round(2)

    agg_funcs = ['mean', 'min', 'max',
                 lambda x: np.percentile(x, 25),
                 lambda x: np.percentile(x, 50),
                 lambda x: np.percentile(x, 75)]

    agg_func_names = ['Mean', 'Min', 'Max', 'Q1', 'Median', 'Q3']

    numeric_cols = [col for col in df.columns if col not in ['Cluster', 'State']]
    agg_dict = {col: agg_funcs for col in numeric_cols}

    summary = df.groupby('Cluster').agg(agg_dict)

    # Format columns: 'recency_mean' => 'Recency Mean'
    summary.columns = [
        f"{col.capitalize()} {stat}" for col in numeric_cols for stat in agg_func_names
    ]

    if debug:
        print("Step 1: Basic Aggregates")
        print(summary.head())

    summary['Number of Customer'] = cluster_size
    summary['Percentage (%)'] = percentages

    if debug:
        print("Step 2: Added Size & Percent")
        print(summary.head())

    # State breakdown
    state_summary = df.groupby('Cluster')['State'].value_counts().unstack().fillna(0)

    if remap_state:
        state_summary.columns = state_summary.columns.map(encoded_to_state)

    if debug:
        print("Step 3: State Summary")
        print(state_summary.head())

    summary = pd.concat([summary, state_summary], axis=1)

    if debug:
        print("Step 4: Final Summary")
        print(summary.head())

    return summary.round(2)

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