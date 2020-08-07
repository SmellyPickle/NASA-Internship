import os
import math

import numpy as np
import netCDF4
import datetime


print(f'Program initialized at {datetime.datetime.now()}')


def write(fname: str, array):
    with open(fname, 'a') as f:
        np.savetxt(f, array, fmt='%.18f')


def load(fname) -> np.ndarray:
    return np.loadtxt(fname)


root = 'D:\\IMERG_TIMESERIES_PYTHON_TXT'
nc4_filenames = []
for dirpath, dirs, files in os.walk('D:\\IMERG_DATA'):
    nc4_filenames.extend([f'{dirpath}\\{f}' for f in files if f.endswith('.nc4')])
min_iteration = None
max_iteration = None
for dirpath, dirs, files in os.walk(root):
    for file in files:
        file_iteration = int(file.split('_')[-1].split('.')[0])
        if min_iteration is None or file_iteration < min_iteration:
            min_iteration = file_iteration
        if max_iteration is None or file_iteration > max_iteration:
            max_iteration = file_iteration
if min_iteration is None:
    start_iteration = 0
else:
    if max_iteration - min_iteration not in (0, 1):
        raise ValueError(f'Iteration problems: min_iteration={min_iteration}, max_iteration={max_iteration}')
    if min_iteration == max_iteration:
        if os.path.isfile(f'{root}\\Lon3599\\Lat1799_{min_iteration}.txt'):
            start_iteration = min_iteration + 1
        elif min_iteration == 0:
            start_iteration = min_iteration
        else:
            raise ValueError(f'min_iteration=max_iteration={min_iteration} but not all 3600 files exist')
    else:
        start_iteration = max_iteration

print(f'Starting on iteration {start_iteration}')

read_times = []
write_times = []
for iteration in range(start_iteration, math.ceil(len(nc4_filenames) / 250)):
    start = iteration * 250
    end = min(iteration * 250 + 250, len(nc4_filenames))
    if os.path.isfile(f'{root}\\Lon3599\\Lat1799_{iteration}.txt') and not os.path.isfile(f'{root}\\Lon3599\\Lat1799_{iteration - 1}.txt'):
        print(f'Skipping iteration {iteration}')
        continue

    time_read_start = datetime.datetime.now()
    print(f'Reading netCDF data: {iteration:02}/{math.ceil(len(nc4_filenames) / 250)}')
    data = np.zeros(shape=(250, 3600, 1800), dtype=np.float32)
    for i, filename in enumerate(nc4_filenames[start:end]):
        with netCDF4.Dataset(filename, 'r') as f:
            precip_cal = np.array(f.variables['precipitationCal'])[0]
        precip_cal[precip_cal == -9999.9] = np.nan
        data[i] = precip_cal
    read_times.append(datetime.datetime.now() - time_read_start)

    time_write_start = datetime.datetime.now()
    time_false_write_end = None
    num_false_write = None
    print(f'Reading data took {time_write_start - time_read_start}, now writing data')
    num_false = 0
    for i in range(3600):
        if iteration == 0:
            os.mkdir(f'{root}\\Lon{i}')
        for j in range(1800):
            if os.path.isfile(f'{root}\\Lon{i}\\Lat{j}_{iteration - 1}.txt'):
                if time_false_write_end is None:
                    time_false_write_end = datetime.datetime.now()
                    num_false_write = i
                if os.path.isfile(f'{root}\\Lon{i}\\Lat{j}_{iteration}.txt'):
                    data_0 = load(f'{root}\\Lon{i}\\Lat{j}_{iteration - 1}.txt')
                    if len(data_0) != 250 * iteration:
                        raise ValueError(f'File {root}\\Lon{i}\\Lat{j}_{iteration - 1}.txt has size {len(data_0)} instead of {250 * iteration}')
                    data_1 = load(f'{root}\\Lon{i}\\Lat{j}_{iteration}.txt')
                    if len(data_1) != 250 * iteration + 250:
                        raise ValueError(f'File {root}\\Lon{i}\\Lat{j}_{iteration}.txt has size {len(data_1)} instead of {250 * iteration + 250}')
                    write(f'{root}\\Lon{i}\\Lat{j}_{iteration}.txt', np.concatenate([data_0, data_1]))
                    os.remove(f'{root}\\Lon{i}\\Lat{j}_{iteration - 1}.txt')
                elif iteration > 0:
                    prev_data = load(f'{root}\\Lon{i}\\Lat{j}_{iteration - 1}.txt')
                    if len(prev_data) != 250 * iteration:
                        raise ValueError(f'{root}\\Lon{i}\\Lat{j}_{iteration - 1}.txt has size {len(prev_data)} instead of {250 * iteration}')
                    write(f'{root}\\Lon{i}\\Lat{j}_{iteration}.txt', np.concatenate([prev_data, data[:, i, j]]))
                    os.remove(f'{root}\\Lon{i}\\Lat{j}_{iteration - 1}.txt')
                else:
                    raise ValueError(f'File that should not exist found: {root}\\Lon{i}\\Lat{j}_{iteration - 1}.txt')
            elif os.path.isfile(f'{root}\\Lon{i}\\Lat{j}_{iteration}.txt'):
                num_false += 1
                continue
            elif iteration == 0:
                write(f'{root}\\Lon{i}\\Lat{j}_{iteration}.txt', data[:, i, j])
            else:
                raise FileNotFoundError(f'{root}\\Lon{i}\\Lat{j}_{iteration - 1}.txt')
        if i % 100 == 0:
            print(f'Finished lon {i}')

    if time_false_write_end is None:
        time_false_write_end = time_write_start
        num_false_write = 0
    write_times.append((datetime.datetime.now() - time_false_write_end, 3600 - num_false_write))
    time_per_write = sum((entry[0] * 3600 / entry[1] for entry in write_times), start=datetime.timedelta(0)) / len(write_times)
    time_per_read = sum(read_times, start=datetime.timedelta(0)) / len(read_times)
    num_remaining = math.ceil(len(nc4_filenames) / 250) - iteration - 1
    print(f'ETA - {datetime.datetime.now() + (time_per_read + time_per_write) * num_remaining}')
