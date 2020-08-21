library(extRemes)
library(fields)
library(foreach)
library(doSNOW)
library(zoo)

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
# clusterMap = t(as.matrix(read.table('D:/no_clusters.txt')))

# clustering_alg = "k-means"
# clusterMap = t(as.matrix(read.table('D:/kmeans_clusters_603030.txt', sep=' '))[llcrnrx:urcrnrx, llcrnry_series:urcrnry_series])

# clustering_alg = 'HDBSCAN 30'
# clusterMap = t(as.matrix(read.table('D:/hdbscan_clusters_30.txt')))

# clustering_alg = 'HDBSCAN 20'
# clusterMap = t(as.matrix(read.table('D:/hdbscan_clusters_20.txt')))

clustering_alg = 'k-means 606060'
clusterMap = t(as.matrix(read.table('D:/kmeans_606060_clusters.txt')))[llcrnrx:urcrnrx, llcrnry_series:urcrnry_series]

print(dim(avePrec))
print(dim(clusterMap))

longitude = seq(llcrnrlon + IMERG_RESOLUTION / 2, urcrnrlon, by=IMERG_RESOLUTION)
latitude = seq(llcrnrlat + IMERG_RESOLUTION / 2, urcrnrlat, by=IMERG_RESOLUTION)

cluster_indexes = unique(as.vector(clusterMap))
num_clusters = length(unique(as.vector(clusterMap)))

# Create text file with run status
# writeLines(c(""), "D:/Algorithm_Status.txt")
start_time  <- Sys.time()
print(start_time)

num_cores = 4
num_per_core = ceiling(length(cluster_indexes) / num_cores)

cl = makeCluster(num_cores, outfile="")
registerDoSNOW(cl)

print(paste0('Total clusters: ', num_clusters))

accum_days = 1

results <- foreach(coreNum=1:num_cores, .combine=cbind, .packages=c("extRemes", "foreach", "zoo"), .verbose=TRUE) %dopar% {
  foreach(iClust=(coreNum * num_per_core - num_per_core + 1):min(coreNum * num_per_core, num_clusters), .combine = cbind) %do% {
    
    if (nrow(which(clusterMap == cluster_indexes[iClust], arr.ind=TRUE)) == 0) return(list(NA, NA, NA, NA, NA, NA, NA, NA))
    # if (nrow(which(clusterMap == cluster_indexes[iClust], arr.ind=TRUE)) == 0) return(NA)
    
    print(paste0('Starting cluster #', iClust, ' of size ', nrow(which(clusterMap == cluster_indexes[iClust], arr.ind=TRUE)),'\n'))

    clustNumber <- cluster_indexes[iClust]
    clust.ind   <- which(clusterMap == clustNumber, arr.ind = TRUE)

    clust.prec <- mean(avePrec[clust.ind], na.rm=TRUE)
    
    main.dir   <- getwd()
    total.data <- list()
    for(i in 1:nrow(clust.ind)){
      lon.i = as.numeric(clust.ind[i,][1]) + llcrnrx - 1
      lat.i = as.numeric(clust.ind[i,][2]) + llcrnry_global - 1
      
      new.dir <- paste0("Lon", lon.i - 1)
      filename = paste0('D:/IMERG_TIMESERIES_PYTHON_GROUPED2/', new.dir, "/Lat", lat.i - 1, ".txt")
      total.data[[i]] <- rollapply(as.matrix(read.table(filename, sep = " "))[,1], accum_days, sum)
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
    
    
    rl.ci = tryCatch({
        fit_MLE = fevd(decl_data, threshold = thresh, type = "GP", method = "MLE", use.phi = TRUE)
        year_list = c(5, 20)
        ci(fit_MLE, type = "return.level", return.period = year_list)
      }, error = function(e) {
        tryCatch({
          print(paste0('Errors in model fitting for cluster ', clustNumber, ', trying Lmoments '))
          fit_Lmom = fevd(decl_data, threshold=thresh, type="GP", method="Lmoments")
          year_list = c(5, 20)
          ci(fit_MLE, type="return.level", return.period = year_list)
        }, error = function(e) {
          print(e)
          print(paste0('Lmoments failed too for cluster, ', clustNumber, ", returning NA"))
          NA
        })
      }
    )
    
    if (all(is.na(rl.ci))) return(list(NA, NA, NA, NA, NA, NA, NA, NA))
    
    # Save parameters
    # If use.phi is TRUE in fit_MLE above, exponentiate the scale parameter
    scale.ind <- exp(fit_MLE$results$par[1]) 
    shape.ind <- fit_MLE$results$par[2]
    
    # Save standard errors of parameter estimates
    st.errors <- sqrt(diag(parcov.fevd(fit_MLE)))
    
    print(paste0('NCR20 = ', (rl.ci[2, 3] - rl.ci[2, 1]) / rl.ci[2, 2]))
    
    # Save results
    result <- list(scale=scale.ind, shape=shape.ind, scale.se=st.errors[1], 
                   shape.se=st.errors[2], threshold=thresh,  sample.size=n_size, 
                   return_level=rl.ci, cluster=clustNumber)
    return(result)
  }
}

stopCluster(cl)
end_time  <- Sys.time()
print(end_time - start_time)

# remove columns with all NA
results = results[, !apply(results, 2, function(x) all(is.na(x)))]

# Save results
scale.fit     <- as.numeric(unlist(results[1,]))
shape.fit     <- as.numeric(unlist(results[2,]))
scale.se      <- as.numeric(unlist(results[3,]))
shape.se      <- as.numeric(unlist(results[4,]))
thresholds    <- as.numeric(unlist(results[5,]))
sample.size   <- as.numeric(unlist(results[6,]))
return_levels <- results[7,]
clust_numbers <- results[8,]

# Compute 95% normalized confidence range
NCR   <- lapply(1:length(return_levels), function(x) as.numeric((return_levels[[x]][,3] - return_levels[[x]][,1]) / return_levels[[x]][,2]))
NCR5  <- unlist(lapply(NCR, function(x) x[1]))
NCR20 <- unlist(lapply(NCR, function(x) x[2]))

# Save scale and shape parameters onto grid
MLE_par <- list()
MLE_par[[1]]   <- matrix(0, nrow = length(longitude), ncol = length(latitude))
MLE_par[[2]]   <- matrix(0, nrow = length(longitude), ncol = length(latitude))
names(MLE_par) <- c("scale", "shape")

# Save scale and shape standard errors onto grid
MLE_ses <- list()
MLE_ses[[1]]   <- matrix(0, nrow = length(longitude), ncol = length(latitude))
MLE_ses[[2]]   <- matrix(0, nrow = length(longitude), ncol = length(latitude))
names(MLE_ses) <- c("scale", "shape")

# Save sample size, thresholds, and NCR onto grid
sample_size_grid <- matrix(0, nrow = length(longitude), ncol = length(latitude))
thresholds_grid  <- matrix(0, nrow = length(longitude), ncol = length(latitude))
NCR5_grid        <- matrix(0, nrow = length(longitude), ncol = length(latitude))
NCR20_grid       <- matrix(0, nrow = length(longitude), ncol = length(latitude))

for(i in 1:length(clust_numbers)){
  cluster_number = clust_numbers[i]
  clust.ind     <- which(clusterMap == cluster_number, arr.ind = TRUE)
  
  # Store parameters
  MLE_par$scale[clust.ind]    <- scale.fit[i]
  MLE_par$shape[clust.ind]    <- shape.fit[i]
  MLE_ses$scale[clust.ind]    <- scale.se[i]
  MLE_ses$shape[clust.ind]    <- shape.se[i]
  sample_size_grid[clust.ind] <- sample.size[i]
  thresholds_grid[clust.ind]  <- thresholds[i]
  NCR5_grid[clust.ind]        <- NCR5[i]
  NCR20_grid[clust.ind]       <- NCR20[i]
}

# my_plot(MLE_par$scale, lon=longitude, lat=latitude, main = "Fitted scale parameters")
# my_plot(MLE_par$shape, main = "Fitted shape parameters")
# my_plot(MLE_ses$scale, main = "Fitted scale standard errors")
# my_plot(MLE_ses$shape, main = "Fitted shape standard errors")
my_plot(NCR5_grid,  lower=0, upper=2, main=paste0("5 year normalized confidence range for ", accum_days, " day accumulations"))
my_plot(NCR20_grid, lower=0, upper=2, main=paste0("20 year normalized confidence range for ", accum_days, " day accumulations"))

ARI_5     <- unlist(lapply(return_levels, function(x) x[1, 2]))
ARI_5_map <- matrix(0, nrow = length(longitude), ncol = length(latitude))

ARI_20     <- unlist(lapply(return_levels, function(x) x[2, 2]))
ARI_20_map <- matrix(0, nrow = length(longitude), ncol = length(latitude))

for(i in 1:length(clust_numbers)){
  clust.ind   <- which(clusterMap == clust_numbers[i], arr.ind = TRUE)
  if(length(clust.ind) > 0) {
    ARI_5_map[clust.ind] = ARI_5[i]
    ARI_20_map[clust.ind] = ARI_20[i]
  }
}

# my_plot(ARI_5_map, main=paste0("5 year ARI for ", accum_days, " day accumulations, ", clustering_alg, " clustering"), lower=min(min(ARI_5_map), min(ARI_20_map)), upper=max(max(ARI_5_map), max(ARI_20_map)))
my_plot(ARI_20_map, main=paste0("20 year ARI for ", accum_days, " day accumulations, ", clustering_alg, " clustering"), lower=min(min(ARI_5_map), min(ARI_20_map)), upper=max(max(ARI_5_map), max(ARI_20_map)))
# write.table(ARI_5_map, file="D:/ARI5_2day_hdsscan.txt", quote=FALSE, row.names=FALSE, col.names=FALSE)
