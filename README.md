# NASA-Internship


| File name                           | Purpose
|-------------------------------------|-----------------------------------------
| `calculate_ncr.R`                   | Calculates entire ARI and NCR maps over a specified range.
| `calculate_percentiles.py`          | Calculates precipitation percentile data.
| `data_conversion.py`                | Converts gridded daily data into timeseries for individual grid points.
| `demirdjian_kmeans_clustering.py`   | Performs a very similar clustering to the one by Dr. Demirdijian.
| `download_data.py`                  | Downloads IMERG NetCDF4 data from NASA.
| `genetic_algorithm.py`              | Tunes k-means clustering parameters to lower average normalized confidence range.
| `modality_testing.R`                | Tests for multimodality in precipitation accumulations.
| `modified_hdbscan.py`               | Performs a somewhat fast, modified version of HDBSCAN that produces much less outliers.
| `sample_ncr.R`                      | Used by `genetic_algorithm.py` to calculate average NCR.
| `view_cluster_size_distribution.py` | Plots cluster size distributions.
| `view_clustering_variables.py`      | Plots clustering variables on a map.
