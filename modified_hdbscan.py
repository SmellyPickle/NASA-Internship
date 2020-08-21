from datetime import datetime
import numpy as np
from scipy.stats import zscore
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
import hdbscan
import matplotlib.pyplot as plt


def exemplars(cluster_id, condensed_tree):
    raw_tree = condensed_tree._raw_tree
    # Just the cluster elements of the tree, excluding singleton points
    cluster_tree = raw_tree[raw_tree['child_size'] > 1]
    # Get the leaf cluster nodes under the cluster we are considering
    leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster_id)
    # Now collect up the last remaining points of each leaf cluster (the heart of the leaf)
    result = np.array([])
    for leaf in leaves:
        max_lambda = raw_tree['lambda_val'][raw_tree['parent'] == leaf].max()
        points = raw_tree['child'][(raw_tree['parent'] == leaf) &
                                   (raw_tree['lambda_val'] == max_lambda)]
        result = np.hstack((result, points))
    return result.astype(np.int)


def dist_vector(point, exemplar_dict, data):
    ret = np.zeros(len(exemplar_dict))
    for reti, cluster in enumerate(exemplar_dict.keys()):
        dists = cdist([data[point]], data[exemplar_dict[cluster].astype(np.int32)])
        ret[reti] = np.amin(dists)
    return ret


def nearest_clust(point, exemplar_dict, data):
    return np.argmin(dist_vector(point, exemplar_dict, data))


# lon = np.linspace(30.05, 39.95, 100)
# lat = np.linspace(-9.95, -0.05, 100)
# average_prec = np.transpose(np.loadtxt('data/average_prec.txt'))[800:900, 2100:2200]
# extreme_prec = np.loadtxt('data/extreme_prec_95.txt')[800:900, 2100:2200]
# topography = np.transpose(np.loadtxt('data/Topography.txt'))[800:900, 2100:2200]

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

start_time = datetime.now()
print(f'Started clustering at {start_time}')

labels_high = MiniBatchKMeans(n_clusters=clust_data.shape[0] // 2000).fit_predict(clust_data[:])
unique_labels = np.unique(labels_high)
print(f'Finished high level k-means clustering')
new_labels = np.zeros_like(labels_high, dtype=int)
for i, label in enumerate(unique_labels):
    mask = (labels_high == label)
    label_add = np.max(new_labels)
    if label_add != 0:
        label_add += 1
    clusterer = hdbscan.HDBSCAN(min_cluster_size=7, min_samples=1, cluster_selection_method='leaf')
    clusterer.fit(clust_data[mask])
    tree = clusterer.condensed_tree_

    exemplar_dict = {c: exemplars(c, tree) for c in tree._select_clusters()}
    clust_subset_labels = np.zeros(clust_data[mask].shape[0])
    for x in range(clust_data[mask].shape[0]):
        clust_subset_labels[x] = nearest_clust(x, exemplar_dict, clust_data[mask])
    new_labels[mask] = clust_subset_labels
    print(f'Finished low level hdbscan for label {i} / {unique_labels.shape[0]}')

labels = new_labels.reshape(1200, 3600)
np.savetxt('data/modified_hdbscan_clusters.txt', labels)

print(f'Clustering took {datetime.now() - start_time}')
print(f'Created {len(np.unique(labels))} clusters, average cluster size is {clust_data.shape[0] / len(np.unique(labels))}')

# swap = np.arange(len(np.unique(labels[labels != -1])))
# np.random.shuffle(swap)
# new_labels = np.zeros(shape=(1200, 3600))
# for i, label in enumerate(np.unique(labels[labels != -1])):
#     new_labels[labels == label] = swap[i]
# new_labels[labels == -1] = -1

fig, axes = plt.subplots(ncols=2, constrained_layout=True)
axes[0].imshow(labels_high.reshape(1200, 3600), origin='lower')
axes[1].imshow(labels, origin='lower', cmap='gist_rainbow')
plt.show()
