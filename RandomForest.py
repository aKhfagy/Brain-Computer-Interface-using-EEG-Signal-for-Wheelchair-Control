from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def RF(X, y):
    l = len(y)
    n_trees = None
    if l <= 10**3:
        n_trees = 300
    elif l <= 10**4:
        n_trees = 400
    else:
        n_trees = 900
    model = RandomForestClassifier(n_estimators=n_trees)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, test_size=0.33,
                                                        random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    error = 0
    percentage = 0
    for i in range(len(predictions)):
        error = error + ((predictions[i] - y_test[i]) ** 2)
        percentage = percentage + (1.0 if predictions[i] == y_test[i] else 0.0)

    percentage = percentage / len(predictions)
    error = np.sqrt(error)

    return model, percentage, error
