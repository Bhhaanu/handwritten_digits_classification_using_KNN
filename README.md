# handwritten_digits_classification_using_KNN

The dataset that was used is hand-written digits images consisting of 10 classes (each class refers to a digit, from 0-9) available on scikit-learn (scikit-learn.org) and already pre-processed. The dataset consists of 1797 instances and 64 attributes and each datapoint is an 8x8 image of integer pixels.
The following are the steps I performed :

Load the dataset:
from sklearn.datasets import load_digits
hw_digits = load_digits()

Visualize the first 5 images. (See the below notes for hints)
Split the data into training and test sets. Use 80% for training and 20% for testing.

Create the KNN classifier:
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors= “Number of neighbors”)

Train the model:
knn_classifier.fit(train_data, train_label)

Calculate the accuracy score:
knn_classifier.score(test_data, test_label)

Perform prediction on test data and print out the misclassified data 
knn_classifier.predict() 

Experiment over multiple k values for KNN and compute the training and testing scores and plot the accuracy. (Y-axis: Model Accuracy, X-axis: K Neighbor)
