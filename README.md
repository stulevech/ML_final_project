# image_classification
Machine learning final project for STOR 565 at UNC. Multi-class image classification of heterogeneous high-resolution maritime vessels.

Feature extraction approaches used include:

1) PC loadings of blue color pixel intensities

2) Image segmentation (Otsu's method) PC loadings

3) PC's after edge detection

4) PC's after data augmentation 

Feature extraction methods were compared for 600 training images following image pre-processing. Pre-processing involved blue color pixel 
extraction, image resizing, and selecting the optimal number of PC components to keep using cross-validation. 

Once PC loadings were prepared for each set of features, a host of applicable classification methods were applied including LDA, QDA, K-NN,
basic decision tree, random forest, XGboosted decision tree, linear and polynomial kernel SVM, and basic neural nets. A convolutional neural net
will be attempted again at another time... 
