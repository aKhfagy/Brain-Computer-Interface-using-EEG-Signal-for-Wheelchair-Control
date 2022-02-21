from FNN import FNN
from RandomForest import RF
from KNN import KNN
from datasets import TUARv2, load_processed_features_TUARv2

features, labels, n_output = load_processed_features_TUARv2()

fuzzy = FNN(features, labels)
fuzzy.make_model(n_inputs=4, n_hidden=4, n_outputs=n_output, n_iterations=100)

knn, knn_accuracy, knn_error = KNN(features, labels, n_output)
print ('Accuracy: ', knn_accuracy, ', Error: ', knn_error)

rf, rf_accuracy, rf_error = RF(features, labels)
print ('Accuracy: ', rf_accuracy, ', Error: ', rf_error)

