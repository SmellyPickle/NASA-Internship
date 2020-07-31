import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

average_prec = np.transpose(np.loadtxt('data/Average_Prec.txt'))[300:900]
extreme_prec = np.transpose(np.loadtxt('data/Extreme_Prec_95.txt'))
topography = np.transpose(np.loadtxt('data/Topography.txt'))[300:900]
lon = np.linspace(-179.95, 179.95, 3600)
lat = np.linspace(-59.95, -0.05, 600)
data_lon, data_lat = np.meshgrid(lon, lat)

data = np.concatenate([
    stats.zscore(data_lon.reshape(-1, 1)),
    stats.zscore(data_lat.reshape(-1, 1)),
    stats.zscore(average_prec.reshape(-1, 1)),
    stats.zscore(extreme_prec.reshape(-1, 1)),
    stats.zscore(topography.reshape(-1, 1))
], axis=1)

num_clusters = (30, 30, 30)
clusters_high = KMeans(n_clusters=num_clusters[0]).fit_predict(data)
np.savez_compressed('data_week_5/kmeans_high.npz', clusters_high)

clusters_med = np.zeros_like(clusters_high)
for i, clust in enumerate(np.unique(clusters_high)):
    result = KMeans(n_clusters=num_clusters[1]).fit_predict(data[clusters_high == clust])
    clusters_med[clusters_high == clust] = result + i * num_clusters[1]

clusters_low = np.zeros_like(clusters_med)
for i, clust in enumerate(np.unique(clusters_med)):
    result = KMeans(n_clusters=num_clusters[2]).fit_predict(data[clusters_med == clust])
    clusters_low[clusters_med == clust] = result + i * num_clusters[2]

fig, axes = plt.subplots(nrows=3)
axes[0].imshow(clusters_high, origin='lower')
axes[1].imshow(clusters_med, origin='lower')
axes[2].imshow(clusters_low, origin='lower')
