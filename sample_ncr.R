library(extRemes)
library(fields)

my_plot <- function(data, numColors=100, lon=longitude, lat=latitude, 
                    lower=min(data, na.rm=TRUE), upper=max(data, na.rm=TRUE), base.white=FALSE, ...) {
  # Define custom color scheme
  if (base.white == TRUE) {
    colfunc <- colorRampPalette(c("white", "blue", "skyblue", "green", "yellow", "red"))
  } else {
    colfunc <- colorRampPalette(c("blue", "skyblue", "green", "yellow", "red"))
  }
  
  image.plot(lon, lat, data,
             breaks = seq(lower, upper, length.out = numColors + 1), 
             col = colfunc(numColors), xlab = "Longitude", ylab = "Latitude",  ...)
  map("world", xlim=range(longitude), ylim=range(latitude), add = TRUE)
}

rolling_sum = function(data, window) {
  total <- length(data)
  spots <- seq(from=1, to=(total-window+1), by=1)
  result <- vector(length = length(spots))
  for(i in 1:length(spots)){
    result[i] <- sum(data[spots[i]:(spots[i]+window-1)])
  }
  return(result)
}


# range of data conversion
timeseries_lat_min = -60

# range of model fitting: "llcrnrlon" is the longitude of the lower left corner. 
# llcrnrlon = 30
# llcrnrlat = -10
# urcrnrlon = 40
# urcrnrlat = 0
llcrnrlon = -180
llcrnrlat = -60
urcrnrlon = 180
urcrnrlat = 60

IMERG_RESOLUTION = 0.1


llcrnrx = (llcrnrlon + 180) / IMERG_RESOLUTION + 1
urcrnrx = (urcrnrlon + 180) / IMERG_RESOLUTION
llcrnry_global = (llcrnrlat + 90) / IMERG_RESOLUTION + 1
urcrnry_global = (urcrnrlat + 90) / IMERG_RESOLUTION
llcrnry_series = (llcrnrlat - timeseries_lat_min) / IMERG_RESOLUTION + 1
urcrnry_series = (urcrnrlat - timeseries_lat_min) / IMERG_RESOLUTION

# Average precipitation data was computed with global range
avePrec    <- as.matrix(read.table('D:/Average_Prec.txt', sep = ' '))[llcrnrx:urcrnrx, llcrnry_global:urcrnry_global]


# clustering_alg = "no"
# cluster_map = t(as.matrix(read.table('D:/no_clusters.txt')))

# clustering_alg = "k-means"
# cluster_map = t(as.matrix(read.table('D:/kmeans_clusters_603030.txt', sep=' '))[llcrnrx:urcrnrx, llcrnry_series:urcrnry_series])

# clustering_alg = 'HDBSCAN 30'
# cluster_map = t(as.matrix(read.table('D:/hdbscan_clusters_30.txt')))

# clustering_alg = 'HDBSCAN 20'
# cluster_map = t(as.matrix(read.table('D:/hdbscan_clusters_20.txt')))

# clustering_alg = 'k-means 606060'
# cluster_map = t(as.matrix(read.table('D:/kmeans_606060_clusters.txt')))[llcrnrx:urcrnrx, llcrnry_series:urcrnry_series]

# clustering_alg = 'k-means precipitation only'
# cluster_map = t(as.matrix(read.table('D:/kmeans_precip_clusters.txt')))[llcrnrx:urcrnrx, llcrnry_series:urcrnry_series]

clustering_alg = "custom"
cluster_map = t(as.matrix(read.table('C:/Users/Jerry Xiong/Desktop/NASA/ga_clusters.txt')))

longitude = seq(llcrnrlon + IMERG_RESOLUTION / 2, urcrnrlon, by=IMERG_RESOLUTION)
latitude = seq(llcrnrlat + IMERG_RESOLUTION / 2, urcrnrlat, by=IMERG_RESOLUTION)

cluster_indexes = unique(as.vector(cluster_map))

start_time  <- Sys.time()
# print(start_time)
# print(paste0('Total clusters: ', length(cluster_indexes)))

accum_days = 1
num_samples = 100
years = 20
cum_NCR = 0
num_points = 0

for (i in 1:num_samples) {
  label = sample(cluster_indexes[cluster_indexes != -1], 1)
  
  if (sum(cluster_map == label) == 0) next
  
  # print(paste0('Starting cluster label ', label, ' of size ', nrow(which(cluster_map == label, arr.ind=TRUE)),'\n'))
  
  clust.ind   <- which(cluster_map == label, arr.ind = TRUE)
  
  clust.prec <- mean(avePrec[clust.ind], na.rm=TRUE)
  
  main.dir   <- getwd()
  total.data <- list()
  for(i in 1:nrow(clust.ind)){
    lon.i = as.numeric(clust.ind[i,][1]) + llcrnrx - 1
    lat.i = as.numeric(clust.ind[i,][2]) + llcrnry_global - 1
    
    new.dir <- paste0("Lon", lon.i - 1)
    filename = paste0('D:/IMERG_TIMESERIES_PYTHON_GROUPED2/', new.dir, "/Lat", lat.i - 1, ".txt")
    total.data[[i]] <- rolling_sum(as.matrix(read.table(filename, sep = " "))[,1], accum_days)
  }
  
  is_valid    <- !is.na(unlist(total.data))
  pooled_data <- unlist(total.data)[is_valid]
  
  # Find threshold. There are 3 cases
  # 1) If a region has very low average precipitation, set a lower
  # bound and only consider precipitation levels above that bound.
  # 2) If a region has only 1 grid point, use the 95th percentile
  # to get some extra data values.
  # 3) Else use the 99th percentile of the pooled series.
  lower_bound <- 0.1
  if(clust.prec <= lower_bound){
    thresh_quant  <- 0.95
    thresh        <- quantile(pooled_data[pooled_data >= lower_bound], probs = thresh_quant)
  }else if((clust.prec > lower_bound) & (nrow(clust.ind) == 1)){
    thresh_quant  <- 0.95
    thresh        <- quantile(pooled_data, probs = thresh_quant)
  }else{
    thresh_quant  <- 0.99
    thresh        <- quantile(pooled_data, probs = thresh_quant)
  }
  
  # Decluster individual series and pool the results
  ind_series <- lapply(1:length(total.data), function(x) total.data[[x]][!is.na(total.data[[x]])])
  decl_data  <- unlist(lapply(1:length(total.data), function(x) decluster(ind_series[[x]], threshold = thresh, r = 5)))
  n_size     <- sum(decl_data >= thresh)
  
  
  fit_MLE = fevd(decl_data, threshold = thresh, type = "GP", method = "MLE", use.phi = TRUE)
  rl.ci = ci(fit_MLE, type = "return.level", return.period = years)
  
  NCR = (rl.ci[3] - rl.ci[1]) / rl.ci[2]
  # print(paste0('NCR', years, ' = ', NCR))
  cum_NCR = cum_NCR + NCR * sum(cluster_map == label)
  num_points = num_points + sum(cluster_map == label)
}

print(unname(cum_NCR) / num_points)
