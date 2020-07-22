import time

import numpy as np
from scipy import stats
import hdbscan
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


avg_prec = np.transpose(np.genfromtxt('D:/Average_Prec.txt'))
# extreme_prec = pd.read_table('D:/Extreme_Prec_95.txt')
topography = np.transpose(np.genfromtxt('D:/Topography.txt'))

x, y = np.linspace(-179.95, 179.95, 3600), np.linspace(-89.95, 89.95, 1800)
xx, yy = np.meshgrid(x, y)

# m = Basemap()
# m.drawcoastlines()
# m.pcolormesh(xx, yy, topography)
# plt.show()

cluster_data = np.concatenate([
    stats.zscore(xx.reshape(-1, 1)),
    stats.zscore(yy.reshape(-1, 1)),
    stats.zscore(avg_prec.reshape(-1, 1)),
    stats.zscore(topography.reshape(-1, 1))
], axis=1)
# cluster_data = pd.DataFrame([xx.flatten(), yy.flatten(), avg_prec.flatten(), topography.flatten()])

clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=1)

start_time = time.time()
cluster_results = clusterer.fit_predict(cluster_data)
end_time = time.time()
print(f'Clustering took {end_time - start_time:.2f} s')

np.savez_compressed('cluster_results_fixed.npz', cluster_results.reshape(1800, 3600))

m = Basemap()
m.drawcoastlines()
m.pcolormesh(xx, yy, cluster_results.reshape(1800, 3600))
plt.show()
