import os
import pandas as pd
import seaborn as sns
import numpy as np
import squarify
from django.contrib.admin import display
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from matplotlib import cm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder
import unidecode
import re
import urllib
from matplotlib import image as mpimg
import folium
from folium.plugins import HeatMap
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from utils import plot_3d_clusters, summarize_cluster, summarize_cluster2
import pickle