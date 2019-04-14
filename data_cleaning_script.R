################### ML Project: Data setup ##########################

### Installing EBImage from BioConductor:
if (!requireNamespace("BiocManager", quietly = TRUE)){
  install.packages("BiocManager")}
BiocManager::install("EBImage", version = "3.8")

### Required packages
library(tidyverse)
library(EBImage)
library(jpeg)
library(reshape2)

### Extracting images for gondola, sailboat, and cruiseship 

gondola.filenames <- list.files(path = "gondola", pattern="*.jpg")
sailboat.filenames <- list.files(path = "sailboat", pattern="*.jpg")
cruiseship.filenames <- list.files(path = "cruise ship", pattern="*.jpg")

gondola.filenames <- gondola.filenames[1:100]
sailboat.filenames <- sailboat.filenames[1:100]
cruiseship.filenames <- cruiseship.filenames[c(1:25, 27:101)] # removing the one PNG image 

setwd("/Users/sumati/Documents/STOR 565/Project/boats/gondola")

gondola.array <-
  array(NA, dim = c(115, 345, length(gondola.filenames)))

for (i in 1:length(gondola.filenames)){
  im <- readJPEG(gondola.filenames[i])
  im <- EBImage::resize(im, w = 115, h = 345) # resize image 
  im <- im[,,1] # isolate pixel intensities 
  gondola.array[,,i] <- im
}

setwd("/Users/sumati/Documents/STOR 565/Project/boats/sailboat")

sailboat.array <-
  array(NA, dim = c(115, 345, length(sailboat.filenames)))

for (i in 1:length(sailboat.filenames)){
  im <- readJPEG(sailboat.filenames[i])
  im <- EBImage::resize(im, w = 115, h = 345) # resize image 
  im <- im[,,1] # isolate pixel intensities 
  sailboat.array[,,i] <- im
}

setwd("/Users/sumati/Documents/STOR 565/Project/boats/cruise ship")


cruiseship.array <-
  array(NA, dim = c(115, 345, length(cruiseship.filenames)))

for (i in 1:length(cruiseship.filenames)){
  im <- readJPEG(cruiseship.filenames[i])
  im <- EBImage::resize(im, w = 115, h = 345) # resize image 
  im <- im[,,1] # isolate pixel intensities 
  cruiseship.array[,,i] <- im
}


setwd("/Users/sumati/Documents/STOR 565/Project/boats")

boats <- matrix(NA, nrow = 300, ncol = 1+(115*345))
boats[,1] <- rep(c(1,2,3), each = 100)
# 1, 2, 3 is gondola, sailboat, cruise

for (i in 1:100){
  melt.array <- melt(gondola.array[,,i])[,3]
  
  for (j in 2:ncol(boats)){
    boats[i,j] <- melt.array[j-1]
  }
}

for (i in 101:200){
  melt.array <- melt(sailboat.array[,,i-100])[,3]
  
  for (j in 2:ncol(boats)){
    boats[i,j] <- melt.array[j-1]
  }
}

for (i in 201:300){
  melt.array <- melt(cruiseship.array[,,i-200])[,3]
  
  for (j in 2:ncol(boats)){
    boats[i,j] <- melt.array[j-1]
  }
}

boats <- as.data.frame(boats)

colnames(boats)[1] <- c("type")

boats$type <- ifelse(boats$type == 1, "gondola", 
                     ifelse(boats$type == 2, "sailboat", "cruise"))

# write_csv(boats, 'boats_pixels.csv')


### Exploratory: how many PCs would be needed? 
# The number of principal components, for each boat image, 
  # that explain 90 percent of the variation in the image’s pixel intensities
components.vec <- numeric(nrow(boats))
for (i in 1:nrow(boats)){
  row <- boats[i, 2:ncol(boats)] %>% as.numeric()
  row.mat <- matrix(row, 
                    nrow = 115, ncol = 345)
  pr.out <- prcomp(row.mat)
  pr.var <- pr.out$sdev^2
  pve <- pr.var / sum(pr.var)
  components.vec[i] <- min(which(cumsum(pve) > 0.9))
}

plot(density(components.vec))
mean(components.vec)
median(components.vec)

# keep 20 components

### Creating PC loadings 
pic1 = boats[1, 2:ncol(boats)] %>% as.numeric()
pic1.mat = matrix(row, nrow = 115, ncol = 345)
pic1.out = prcomp(pic1.mat)
loadings = pic1.out$rotation
loadings = loadings[, 1:20]


loadings.array = array(NA, dim = c(345, 20, 300))
for (i in 1:nrow(boats)) {
  pic = boats[i, 2:ncol(boats)] %>% as.numeric()
  pic.mat = matrix(row, nrow = 115, ncol = 345)
  pic.out = prcomp(pic.mat)
  loadings = pic.out$rotation
  loadings = loadings[, 1:20]
  loadings.array[,,i] = loadings 
}



