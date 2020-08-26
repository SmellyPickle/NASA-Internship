"""Performs a very similar clustering to the original recursive k-means scheme by Dr. Demirdjian

There are some notable differences. Firstly, the number of clusters is 60, 60, 60 at high, medium, and low levels.
(This can be changed easily. These values are set to what they are so as to have approximately the same number of 
points per cluster). Secondly, it uses mini-batch k-means, which is much faster at the cost of a very slight 
difference in the final results. Finally, it uses 95th percentile data instead of 90th.
"""

import numpy as np
from scipy.stats import zscore
from sklearn.cluster import MiniBatchKMeans

num_clusters = (60, 60, 60)

lon = np.linspace(-179.95, 179.95, 3600)
lat = np.linspace(-59.95, 59.95, 1200)
average_prec = np.transpose(np.loadtxt('data/average_prec.txt'))[300:1500]
extreme_prec = np.loadtxt('data/extreme_prec_95.txt')[300:1500]
topography = np.transpose(np.loadtxt('data/Topography.txt'))[300:1500]

lon_data, lat_data = np.meshgrid(lon, lat)
assert lon_data.shape == lat_data.shape == average_prec.shape == extreme_prec.shape == topography.shape == (1200, 3600)
clust_data = np.concatenate([
    zscore(lon_data.reshape(-1, 1)),
    zscore(lat_data.reshape(-1, 1)),
    zscore(average_prec.reshape(-1, 1)),
    zscore(extreme_prec.reshape(-1, 1)),
    zscore(topography.reshape(-1, 1))
], axis=1)

labels = np.zeros(clust_data.shape[0], dtype=int)
for k in num_clusters:
    print(f'Starting clustering stage with {k} centers')
    new_labels = np.zeros_like(labels, dtype=int)
    for label in np.unique(labels):
        mask = (labels == label)
        sz = np.count_nonzero(mask)
        clust_subset_data = clust_data[mask]
        if sz == 0:
            continue
        elif sz < k:
            subset_labels = np.zeros(sz)
        else:
            subset_labels = MiniBatchKMeans(n_clusters=k).fit_predict(clust_subset_data)
        if np.count_nonzero(labels != label) > 0:
            subset_labels += np.max(new_labels) + 1
        new_labels[labels == label] = subset_labels
    labels = new_labels
labels = labels.reshape(1200, 3600)
np.savetxt(f'data/kmeans_{"".join(str(x) for x in num_clusters)}_clusters.txt', labels)

print(f'Created {len(np.unique(labels))} clusters, average cluster size is {clust_data.shape[0] / len(np.unique(labels))}')
