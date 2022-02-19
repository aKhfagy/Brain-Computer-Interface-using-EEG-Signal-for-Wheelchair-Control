from sklearn.neighbors import KNeighborsClassifier

def KNN(X, y, k):
    model = KNeighborsClassifier(n_neighbors=k)
