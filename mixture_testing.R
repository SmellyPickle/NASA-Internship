library(diptest)
library(extRemes)
library(fields)
library(zoo)

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
avePrec    = as.matrix(read.table('D:/Average_Prec.txt', sep = ' '))[llcrnrx:urcrnrx, llcrnry_global:urcrnry_global]

longitude = seq(llcrnrlon + IMERG_RESOLUTION / 2, urcrnrlon, by=IMERG_RESOLUTION)
latitude = seq(llcrnrlat + IMERG_RESOLUTION / 2, urcrnrlat, by=IMERG_RESOLUTION)


# num_mix2_better_AIC = 0
# num_mix3_better_AIC = 0
# num_mix2_better_BIC = 0
# num_mix3_better_BIC = 0

# for (i in 1:length(longitude)) {
#   for (j in 1:length(latitude)) {
#     filename = paste0("D:/IMERG_TIMESERIES_PYTHON_GROUPED2/Lon", i + llcrnrx - 2, "/Lat", j + llcrnry_global - 2, ".txt")
#     gridpoint_data = as.vector(read.table(filename, sep=" "))
#     gridpoint_data = rollapply(gridpoint_data[!is.na(gridpoint_data)], accum_days, sum)
#     # if (dip.test(gridpoint_data)$p.value < 0.1) print(paste("p =", dip.test(gridpoint_data)$p.value, "at", i, j))
#     if (dip.test(gridpoint_data[gridpoint_data > 0])$p.value < 0.1) print(paste("p =", dip.test(gridpoint_data[gridpoint_data > 0])$p.value, "at", i + llcrnrx - 1, j + llcrnry_global - 1, "(1-indexed) for >0 values"))
#     # print(dip.test(gridpoint_data)$p.value)
#     # print(dip.test(gridpoint_data[gridpoint_data > 0])$p.value)
#     # mix2 = normalmixEM(gridpoint_data[gridpoint_data > 0], k=2, maxrestarts=1000)
#     # mix3 = normalmixEM(gridpoint_data[gridpoint_data > 0], k=3, maxrestarts=1000)
#     # 
#     # clust.prec = avePrec[i][j]
#     # lower_bound = 0.1
#     # if(is.na(clust.prec) || clust.prec <= lower_bound){
#     #   thresh_quant = 0.95
#     #   thresh = quantile(gridpoint_data[gridpoint_data >= lower_bound], probs=thresh_quant)
#     # } else {
#     #   thresh_quant = 0.95
#     #   thresh = quantile(gridpoint_data, probs=thresh_quant)
#     # }
#     # ind_series = lapply(1:length(gridpoint_data), function(x) gridpoint_data[[x]][!is.na(gridpoint_data[[x]])])
#     # decl_data  = unlist(lapply(1:length(gridpoint_data), function(x) decluster(ind_series[[x]], threshold=thresh, r=5)))
#     # n_size     = sum(decl_data >= thresh)
#     # fit_MLE = fevd(decl_data, threshold=thresh, type="GP", method="MLE", use.phi=TRUE)
#     # 
#     # gp_sum = summary(fit_MLE)
#     # mix2_AIC = 2*(3*2 - 1) - 2*mix2$loglik
#     # mix3_AIC = 2*(3*3 - 1) - 2*mix3$loglik
#     # gp_AIC = gp_sum$AIC
#     # mix2_BIC = log(length(gridpoint_data[gridpoint_data > 0]))*(3*2 - 1) - 2*mix2$loglik
#     # mix3_BIC = log(length(gridpoint_data[gridpoint_data > 0]))*(3*3 - 1) - 2*mix3$loglik
#     # gp_BIC = gp_sum$BIC
#     # 
#     # if (mix2_AIC < gp_AIC) num_mix2_better_AIC = num_mix2_better_AIC + 1
#     # if (mix3_AIC < gp_AIC) num_mix3_better_AIC = num_mix3_better_AIC + 1
#     # if (mix2_BIC < gp_BIC) num_mix2_better_BIC = num_mix2_better_BIC + 1
#     # if (mix3_BIC < gp_BIC) num_mix3_better_BIC = num_mix3_better_BIC + 1
#   }
# }

labels = t(as.matrix(read.table('D:/kmeans_606060_clusters.txt')))[llcrnrx:urcrnrx, llcrnry_series:urcrnry_series]
# labels = matrix(1:(3600 * 1200), nrow=3600, ncol=1200)

accum_days = 7
num_significant = 0
num_tried = 0
for (label in sample(1:max(labels))) {
  if (nrow(which(labels == label, arr.ind=TRUE)) == 0) next
  
  # print(paste0('Starting cluster #', iClust, ' of size ', nrow(which(labels == label, arr.ind=TRUE)),'\n'))
  
  clust.ind   <- which(labels == label, arr.ind=TRUE)
  
  clust.prec <- mean(avePrec[clust.ind], na.rm=TRUE)
  
  main.dir   <- getwd()
  total.data <- list()
  for(i in 1:nrow(clust.ind)){
    lon.i = as.numeric(clust.ind[i,][1]) + llcrnrx - 1
    lat.i = as.numeric(clust.ind[i,][2]) + llcrnry_global - 1
    
    new.dir <- paste0("Lon", lon.i - 1)
    filename = paste0('D:/IMERG_TIMESERIES_PYTHON_GROUPED2/Lon', lon.i - 1, "/Lat", lat.i - 1, ".txt")
    total.data[[i]] <- rollapply(as.matrix(read.table(filename, sep = " "))[,1], accum_days, sum)
  }
  
  is_valid    <- !is.na(unlist(total.data))
  pooled_data <- unlist(total.data)[is_valid]
  
  a = 0.05
  
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

  pot_dip = dip.test(decl_data[decl_data > thresh])$p.value

  if (pot_dip < a) {
    num_significant = num_significant + 1
    print(paste(label, 'pot', pot_dip))
  }
  
  num_tried = num_tried + 1
  print(paste(num_significant, '/', num_tried, 'were significant at the a =', a, 'level'))
}

print(num_significant / num_tried)

