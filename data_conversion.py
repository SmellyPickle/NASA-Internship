import os
import math

import numpy as np
import netCDF4
from datetime import datetime

print(f'Program initialized at {datetime.now()}')

root = 'D:\\IMERG_TIMESERIES_PYTHON_GROUPED2'
nc4_filenames = []
for dirpath, dirs, files in os.walk('D:\\IMERG_DATA'):
    nc4_filenames.extend([f'{dirpath}\\{f}' for f in files if f.endswith('.nc4')])

groups = [(0, 300), (900, 1200), (1200, 1500), (1500, 1800)]  # latitude values
iter_size = 2000
for group in groups:
    for iteration in range(math.ceil(len(nc4_filenames) / iter_size)):
        read_start = datetime.now()
        sz = iter_size if (iteration + 1) * iter_size <= len(nc4_filenames) else len(nc4_filenames) % iter_size
        data = np.zeros(shape=(sz, 3600, 300), dtype=np.float32)
        for i, filename in enumerate(nc4_filenames[iteration * iter_size:(iteration + 1) * iter_size]):
            with netCDF4.Dataset(filename, 'r') as f:
                precip_cal = np.array(f.variables['precipitationCal'])[0, :, group[0]:group[1]]
            precip_cal[precip_cal == -9999.9] = np.nan
            data[i] = precip_cal
        print(f'Reading completed - group {group} iteration {iteration} - Elapsed {datetime.now() - read_start}')

        write_start = datetime.now()
        for lon in range(3600):
            if iteration == 0 and group == groups[0]:
                os.mkdir(f'{root}\\Lon{lon}')
            for lat in range(group[0], group[1]):
                with open(f'{root}\\Lon{lon}\\Lat{lat}.txt', 'a') as f:
                    f.write(''.join(str(x) + '\n' for x in data[:, lon, lat - group[0]]))
        print(f'Writing completed - group {group} iteration {iteration} - Elapsed {datetime.now() - write_start}')
