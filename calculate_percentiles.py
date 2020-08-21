import numpy as np


percentiles = [25, 50, 75, 95]

root = 'D:/IMERG_TIMESERIES_PYTHON_GROUPED2'
results = [np.zeros(shape=(1800, 3600)) for _ in range(4)]
for lon in range(3600):
    for lat in range(1800):
        data = np.loadtxt(f'{root}/Lon{lon}/Lat{lat}.txt')
        if np.count_nonzero(~np.isnan(data)) == 0:
            print(f'Lon {lon} Lat {lat} is all nan')
            for i in range(4):
                results[i][lat, lon] = np.nan
        else:
            for i, q in enumerate(percentiles):
                results[i][lat, lon] = np.nanpercentile(data, q)
            print(f'Finished Lon {lon} Lat {lat}')

for i, percentile in enumerate(percentiles):
    np.savetxt(f'extreme_prec_{percentile}.txt', results[i])
