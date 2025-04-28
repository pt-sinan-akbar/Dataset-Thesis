import pickle
import pandas as pd
from kmodes.kprototypes import KPrototypes
from utils import plot_3d_clusters, summarize_cluster

# Custom for python script
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

with open('../05-outlier/rfmd_clean.pkl', 'rb') as file:
    df_clean = pickle.load(file)
with open('../08-rfmd-final-processing/encoded_to_state.pkl', 'rb') as file:
    encoded_to_state = pickle.load(file)
with open('../08-rfmd-final-processing/state_mapping.pkl', 'rb') as file:
    state_mapping = pickle.load(file)
with open('../08-rfmd-final-processing/rfmd_final.pkl', 'rb') as file:
    RFMD_final = pickle.load(file)

# END

# Setelah dapet cluster yg bagus dari atas
best_k = 4

# imported from huang
rfmd_kproto = pd.DataFrame(RFMD_final, columns=['recency', 'frequency', 'monetary', 'State'])
gamma_value = 1.0
categorical_indices = [3]
# end

kproto = KPrototypes(n_clusters=best_k, init='Huang', gamma=gamma_value)
clusters_kproto = kproto.fit_predict(rfmd_kproto, categorical=categorical_indices)

rfmd_kproto['Cluster'] = clusters_kproto

print(rfmd_kproto.head())

print("raw rfmd count:", len(df_clean))
print("kproto cluster count:", len(kproto.labels_))
kproto_rfmd_raw = pd.DataFrame(df_clean, columns=["recency", "frequency", "monetary", "State"])
kproto_rfmd_raw['Cluster'] = kproto.labels_
print(kproto_rfmd_raw.head())

# summary
print("Cluster summary:")
print(summarize_cluster(rfmd_kproto))

# print("KProto cluster summary (Original Data):")
print("Cluster summary (Original Data):")
print(summarize_cluster(kproto_rfmd_raw, False))

# plot 3d clusters
plot_3d_clusters(kproto_rfmd_raw, "K-Prototypes")