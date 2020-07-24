import datetime
import os
os.environ['PROJ_LIB'] = 'C:/Users/Jerry Xiong/miniconda3/pkgs/basemap-1.3.0-py38hcdd3ad8_2/Library/share/basemap'

from mpl_toolkits.basemap import Basemap
import netCDF4
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def date_range(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


# TODO: Once data for 2020-06-30 drops, add it.
means = []
# for d in date_range(datetime.date(2000, 6, 1), datetime.date(2020, 6, 30)):
#     with netCDF4.Dataset(f'D:/IMERG_DATA/{d.year}/{d.month}/{d.day}.nc4', 'r') as f:
#         precip_cal = np.array(f.variables['precipitationCal'])[0]
#         precips: np.ndarray = precip_cal.reshape(-1)
#         print(np.count_nonzero(precips > 0) / len(precips))
#         print(np.percentile(precips[precips > 0], 1))
#         print(np.percentile(precips[precips > 0], 10))
#         print(np.percentile(precips[precips > 0], 50))
#         print(np.percentile(precips[precips > 0], 90))
#         # plt.hist(precips, bins='auto')
#         break

d = datetime.date(2004, 6, 6)
with netCDF4.Dataset(f'D:/IMERG_DATA/{d.year}/{d.month:02}/{d.day:02}.nc4', 'r') as f:
    precip_cal = np.array(f.variables['precipitationCal'])[0]
precip_cal[precip_cal > 20] = 20
print(np.min(precip_cal))
precip_cal[precip_cal == -9999.9] = np.nan
print(np.min(precip_cal))

# my_map = Basemap()
# my_map.drawcoastlines()
# my_map.fillcontinents(color='moccasin', lake_color='lightblue')
# my_map.drawmapboundary(fill_color='pink')

plt.pcolormesh(np.transpose(precip_cal))
plt.show()
# print(np.min(precip_cal))
# print(np.max(precip_cal))
# print(np.mean(precip_cal))
