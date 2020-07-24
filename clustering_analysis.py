import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

average_prec = np.transpose(np.loadtxt('D:/Average_Prec.txt'))[300:900]
# extreme_prec = np.transpose(np.loadtxt('D:/Extreme_Prec_95.txt'))
# topography = np.transpose(np.loadtxt('D:/Topography.txt'))[300:900]
# clustering_results = np.transpose(np.loadtxt('D:/Clustering_Results.txt'))

# my_series = pd.Series(clustering_results.flatten())
# counts = my_series.value_counts()

fig, axes = plt.subplots(nrows=2)
axes[0].imshow(average_prec, origin='lower')
average_prec[500:600, 2100:2200] = 0
axes[1].imshow(average_prec, origin='lower')
plt.show()

# fig, ax = plt.subplots()
# ax.hist(counts, bins=30, log=True)
# ax.set_xlabel('Cluster Size')
# ax.set_ylabel('Frequency')
# plt.show()

# fig, axes = plt.subplots(nrows=4)
# axes[0].imshow(average_prec, origin='lower')
# axes[1].imshow(extreme_prec, origin='lower')
# axes[2].imshow(topography, origin='lower')
# axes[3].imshow(clustering_results, origin='lower')


# print(average_prec[clustering_results == 1])

# fig, axes = plt.subplots(nrows=4, ncols=2)
# axes[0, 0].set_title('Average Precipitation')
# axes[0, 0].imshow(average_prec, origin='lower')
#
# axes[1, 0].set_title('95 Percentile Precipitation')
# axes[1, 0].imshow(extreme_prec, origin='lower')
#
# axes[2, 0].set_title('Topography')
# axes[2, 0].imshow(topography, origin='lower')
#
# # axes[3, 0].set_title('Topography (log scale)')
# # axes[3, 0].imshow(topography, norm=LogNorm(vmin=), origin='lower')
#
# axes[3, 0].set_title('Clusters')
# axes[3, 0].imshow(clustering_results, origin='lower')
#
# axes[0, 1].set_title('Small (<=25) Clusters')
# axes[0, 1].imshow(np.isin(clustering_results, counts.index[counts <= 25]), origin='lower')
#
# axes[1, 1].set_title('Small (<=25) and Medium (<=250) Clusters')
# axes[1, 1].imshow(2 * np.isin(clustering_results, counts.index[counts <= 25]) + np.isin(clustering_results, counts.index[np.logical_and(counts > 25, counts <= 250)]), origin='lower')
#
# axes[2, 1].set_title('Large (>=1000) Clusters')
# axes[2, 1].imshow(np.isin(clustering_results, counts.index[counts >= 1000]), origin='lower')
#
# axes[3, 1].set_title('Clusters by size (log scale)')
# im = axes[3, 1].imshow(np.array(counts[clustering_results.flatten()]).reshape(600, 3600), norm=LogNorm(), origin='lower')
# # fig.colorbar(im, ax=axes[3, 1], fraction=0.05, shrink=0.5)
#
#
# def on_resize(event):
#     fig.tight_layout()
#     fig.canvas.draw()
#
#
# fig.canvas.mpl_connect('resize_event', on_resize)
#
# plt.show()
