"""Plots cluster size distributions"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

labels = np.loadtxt('data/ga_insp_clusters.txt')
counts = pd.Series(labels.flatten()).value_counts()
fig, axes = plt.subplots(ncols=2, constrained_layout=True)

axes[0].set_title('Cluster size distribution')
try:
    bins = min((x for x in range(min(20, np.ptp(counts) + 1), 60) if (np.ptp(counts) + 1) % x == 0),
               key=lambda x: abs(x - 30))
except ValueError:
    bins = 30
axes[0].hist(counts, bins=bins)

axes[1].set_title('Cluster size distribution (small clusters)')
axes[1].hist(counts[counts <= 50], bins=np.max(counts[counts <= 50] - np.min(counts[counts <= 50] + 1)))

plt.show()
