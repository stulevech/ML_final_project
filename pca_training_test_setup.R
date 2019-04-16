### PCA and the Creation of Training, Testing Sets ###
### Use results for LDA, SVM's, KNN's and Neural Networks ###

boats_pixels = read.csv('boats_pixels.csv')

#Remove class labels prior to PCA
boats_labels = boats_pixels$type
boats_pixels = boats_pixels %>% select(-type)
pca.out = prcomp( boats_pixels, scale = T )

#Add boat type class labels back as "class"
pca.result = data.frame(class = boats_labels, pca.out$x[,1:20])

#Separate into 80% train, 20% validate
train_ids <- sample((1:300), size = floor(.80*300))
train = pca.result[train_ids,]
validate = pca.result[-train_ids,]