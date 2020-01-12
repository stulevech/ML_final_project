#image_classification

This Machine Learning (STOR 565) group project at UNC-Chapel Hill explores the strengths of several feature extraction methods and classification methods to build a practical multi-class classification algorithm for high-resolution boat images. 

Data sources: Kaggle dataset of 1500 high-resolution boat images, collected and organized by Pixabay:
              https://www.kaggle.com/clorichel/boat-types-recognition/version/1
              Three 200-boat samples are taken from from each of three image classes: gondola, sailboat, and cruise ship.

Feature extraction approaches used include:
  1) PC loadings of blue color pixel intensities
  2) Image segmentation (Otsu's method) PC loadings
  3) PC's following edge detection 
  4) PC's following data augmentation 

Feature extraction methods were compared for 600 training images following image pre-processing. Pre-processing involved blue color pixel 
extraction, image resizing, and selecting the optimal number of PC components to keep using cross-validation. 

Once PC loadings were prepared for each set of features, a host of applicable classification methods were applied including LDA, QDA, K-NN,
basic decision tree, random forest, XGboosted decision tree, linear and polynomial kernel SVM, and basic neural nets. A convolutional neural net will be attempted again at another time... 
