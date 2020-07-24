import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
#
# average_prec = np.transpose(np.loadtxt('D:/Average_Prec.txt'))[300:900]
# extreme_prec = np.transpose(np.loadtxt('D:/Extreme_Prec_95.txt'))
# topography = np.transpose(np.loadtxt('D:/Topography.txt'))[300:900]
# cluster_labels = np.transpose(np.loadtxt('D:/Clustering_Results.txt'))
#
# my_series = pd.Series(cluster_labels.flatten())
# counts = my_series.value_counts()
#
# x = np.zeros(len(counts.index))
# y1 = np.zeros_like(x)
# y2 = np.zeros_like(x)
# y3 = np.zeros_like(x)
#
# for i, clust in enumerate(counts.index):
#     x[i] = (counts[clust])
#     points_in_cluster = cluster_labels == clust
#     y1[i] = np.nanmean(average_prec[points_in_cluster])
#     y2[i] = np.nanmean(extreme_prec[points_in_cluster])
#     y3[i] = np.nanmean(topography[points_in_cluster])
#
# np.savez_compressed('cluster_size_analysis_data.npz', x=x, y1=y1, y2=y2, y3=y3)

data = np.load('cluster_size_analysis_data.npz')
x, y1, y2, y3 = data['x'], data['y1'], data['y2'], data['y3']

fig, axes = plt.subplots(ncols=3)

avg_s = y1[x <= 25]
avg_m = y1[np.logical_and(x > 25, x <= 250)]
avg_l = y1[x > 250]

extr_s = y2[x <= 25]
extr_m = y2[np.logical_and(x > 25, x <= 250)]
extr_l = y2[x > 250]

topo_s = y3[x <= 25]
topo_m = y3[np.logical_and(x > 25, x <= 250)]
topo_l = y3[x > 250]

axes[0].set_title('Average Precipitation')
axes[0].boxplot([avg_s, avg_m, avg_l], sym='.')
axes[0].set_xticklabels(('Small (n<=25)', 'Medium (25<n<=250)', 'Large (250<n)'))
axes[0].set_xlabel('Cluster Size')
axes[0].set_ylabel('Precipitation (mm)')

axes[1].set_title('95th Percentile Precipitation')
axes[1].boxplot([extr_s, extr_m, extr_l], sym='.')
axes[1].set_xticklabels(('Small (n<=25)', 'Medium (25<n<=250)', 'Large (250<n)'))
axes[1].set_xlabel('Cluster Size')
axes[1].set_ylabel('Precipitation (mm)')

axes[2].set_title('Topography')
axes[2].boxplot([topo_s, topo_m, topo_l], sym='.')
axes[2].set_xticklabels(('Small (n<=25)', 'Medium (25<n<=250)', 'Large (250<n)'))
axes[2].set_xlabel('Cluster Size')

# fig, axes = plt.subplots(nrows=2, ncols=2)
# 
# axes[0, 0].set_title('Cluster Size vs Average Precipitation')
# axes[0, 0].scatter(x, y1, s=1)
# 
# axes[1, 0].set_title('Cluster Size vs 95th Percentile Precipiation')
# axes[1, 0].scatter(x, y2, s=1)
# 
# axes[0, 1].set_title('Cluster Size vs Average Precipitation, small (<300 grid points) clusters only')
# axes[0, 1].scatter(x[x < 300], y1[x < 300], s=1)
# 
# axes[1, 1].set_title('Cluster Size vs 95th Percentile Precipiation, small (<300 grid points) clusters only')
# axes[1, 1].scatter(x[x < 300], y2[x < 300], s=1)


def on_resize(event):
    fig.tight_layout()
    fig.canvas.draw()


fig.canvas.mpl_connect('resize_event', on_resize)

plt.show()
