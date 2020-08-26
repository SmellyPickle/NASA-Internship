"""Plots clustering variables on a map"""

import numpy as np
import matplotlib.pyplot as plt

lon = np.linspace(30.05, 39.95, 100)
lat = np.linspace(-9.95, -0.05, 100)
lon_data, lat_data = np.meshgrid(lon, lat)
prec_avg = np.transpose(np.loadtxt('data/average_prec.txt'))[800:900, 2100:2200]
prec_50th = np.loadtxt('data/extreme_prec_50.txt')[800:900, 2100:2200]
prec_95th = np.loadtxt('data/extreme_prec_95.txt')[800:900, 2100:2200]
topography = np.transpose(np.loadtxt('data/Topography.txt'))[800:900, 2100:2200]

fig, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True)

axes[0, 0].set_title('Longitude')
axes[0, 0].imshow(lon_data.reshape(100, 100), origin='lower')
axes[0, 1].set_title('Latitude')
axes[0, 1].imshow(lat_data.reshape(100, 100), origin='lower')
axes[0, 2].set_title('Topography')
axes[0, 2].imshow(topography, origin='lower')
axes[1, 0].set_title('Average Precipitation')
axes[1, 0].imshow(prec_avg, origin='lower')
axes[1, 1].set_title('50th Percentile Precipitation')
axes[1, 1].imshow(prec_50th, origin='lower')
axes[1, 2].set_title('95th Percentile Precipitation')
axes[1, 2].imshow(prec_95th, origin='lower')

plt.show()
