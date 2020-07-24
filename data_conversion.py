import os
import math

import numpy as np
import netCDF4
import datetime


# def date_range(start_date, end_date):
#     for n in range(int((end_date - start_date).days)):
#         yield start_date + datetime.timedelta(n)


# def process_files(name, filenames):
#     print(f'Thread {name} starting with {len(filenames)} files')
#     data = np.zeros(shape=(250, 3600, 1800))
#     for i, filename in enumerate(filenames):
#         print(f'Thread {name} reading {filename} - {i+1:02}/{len(filenames)}')
#         with netCDF4.Dataset(filename, 'r') as f:
#             precip_cal = np.array(f.variables['precipitationCal'])[0]
#         precip_cal[precip_cal == -9999.9] = np.nan
#         data[i] = precip_cal
#
#     for i in range(3600):
#         if iteration == 0:
#             os.mkdir(f'{root}\\Lon{i}')
#         print(f'Thread {name} processing longitude {i+1:04}/3600')
#         for j in range(1800):
#             np.savez_compressed(f'{root}\\Lon{i}\\Lat{j}_{name}.npz', data[:, i, j])
#     return f'Thread {name} completed'


root = 'C:\\Users\\Jerry Xiong\\Desktop\\IMERG_TIMESERIES_PYTHON'

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
        if os.path.isfile(f'{root}\\Lon3599\\Lat1799_{min_iteration}.npz'):
            start_iteration = min_iteration + 1
        elif min_iteration == 0:
            start_iteration = min_iteration
        else:
            raise ValueError(f'min_iteration=max_iteration={min_iteration} but not all 3600 files exist')
    else:
        start_iteration = max_iteration

print(f'Starting on iteration {start_iteration}')

# print(f'Files: {nc4_filenames}')

iteration_times = []
for iteration in range(start_iteration, math.ceil(len(nc4_filenames) / 250)):
    start = iteration * 250
    end = min(iteration * 250 + 250, len(nc4_filenames))
    if os.path.isfile(f'{root}\\Lon3599\\Lat1799_{iteration}.npz') and not os.path.isfile(f'{root}\\Lon3599\\Lat1799_{iteration - 1}.npz'):
        print(f'Skipping iteration {iteration}')
        continue

    start_time = datetime.datetime.now()
    data = np.zeros(shape=(250, 3600, 1800), dtype=np.float32)
    print(f'Reading netCDF data: {iteration:02}/{math.ceil(len(nc4_filenames) / 250)}')
    for i, filename in enumerate(nc4_filenames[start:end]):
        with netCDF4.Dataset(filename, 'r') as f:
            precip_cal = np.array(f.variables['precipitationCal'])[0]
        precip_cal[precip_cal == -9999.9] = np.nan
        data[i] = precip_cal

    print(f'Writing npz data')
    for i in range(3600):
        if iteration == 0:
            os.mkdir(f'{root}\\Lon{i}')
        for j in range(1800):
            if os.path.isfile(f'{root}\\Lon{i}\\Lat{j}_{iteration - 1}.npz'):
                if os.path.isfile(f'{root}\\Lon{i}\\Lat{j}_{iteration}.npz'):
                    data_0 = np.load(f'{root}\\Lon{i}\\Lat{j}_{iteration - 1}.npz')['arr_0']
                    try:
                        data_1 = np.load(f'{root}\\Lon{i}\\Lat{j}_{iteration}.npz')['arr_0']
                    except ValueError:
                        print(f'Problems with file {root}\\Lon{i}\\Lat{j}_{iteration}.npz, rewriting data')
                        np.savez_compressed(f'{root}\\Lon{i}\\Lat{j}_{iteration}.npz', np.concatenate([data_0, data[:, i, j]]))
                        os.remove(f'{root}\\Lon{i}\\Lat{j}_{iteration - 1}.npz')
                    else:
                        np.savez_compressed(f'{root}\\Lon{i}\\Lat{j}_{iteration}.npz', np.concatenate([data_0, data_1]))
                        os.remove(f'{root}\\Lon{i}\\Lat{j}_{iteration - 1}.npz')
                elif iteration >= 1:
                    prev_data = np.load(f'{root}\\Lon{i}\\Lat{j}_{iteration - 1}.npz')['arr_0']
                    np.savez_compressed(f'{root}\\Lon{i}\\Lat{j}_{iteration}.npz', np.concatenate([prev_data, data[:, i, j]]))
                    os.remove(f'{root}\\Lon{i}\\Lat{j}_{iteration - 1}.npz')
                else:
                    raise ValueError(f'File that should not exist found: {root}\\Lon{i}\\Lat{j}_{iteration - 1}.npz')
            elif os.path.isfile(f'{root}\\Lon{i}\\Lat{j}_{iteration}.npz'):
                continue
            elif iteration == 0:
                np.savez_compressed(f'{root}\\Lon{i}\\Lat{j}_{iteration}.npz', data[:, i, j])
            else:
                raise FileNotFoundError(f'{root}\\Lon{i}\\Lat{j}_{iteration - 1}.npz')
        if i % 100 == 0:
            print(f'Finished lon {i}')

    iteration_times.append(datetime.datetime.now() - start_time)
    time_per_iteration = sum(iteration_times, start=datetime.timedelta()) / len(iteration_times)
    num_remaining = math.ceil(len(nc4_filenames) / 250) - iteration - 1
    print(f'ETA - {datetime.datetime.now() + time_per_iteration * num_remaining}')


# with concurrent.futures.ThreadPoolExecutor() as executor:
#     futures = []
#     for iteration in range(math.ceil(len(nc4_filenames) / 250)):
#         start = iteration * 250
#         end = min(iteration * 250 + 250, len(nc4_filenames))
#         futures.append(executor.submit(process_files, iteration, nc4_filenames[start:end]))
#     for future in concurrent.futures.as_completed(futures):
#         print(future.result())
