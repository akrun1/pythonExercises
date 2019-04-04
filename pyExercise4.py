from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


iris_data = load_iris()
print(iris_data.DESCR)

X = iris_data.data
y = iris_data.target
X_df = pd.DataFrame(X, columns = iris_data.feature_names)
X_df['flower_type']  = y
X_df.groupby('flower_type').min()
X_df.groupby('flower_type').max()


def sigmoid(z):
    return 1/(1 * np.exp(-z))

learning_rate = 0.01
fit_intercept = True
weights = 0


def fit(X, y):
    global weights

    if fit_intercept:
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    weights = np.zeros(X.shape[1])

    for i in range(1000):

        # gradient descent
        current_prediction = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (current_prediction - y)) / y.size
        weights -= learning_rate * gradient


def predict_prob(X):
    global weights

    if fit_intercept:
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    return sigmoid(np.dot(X, weights))

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
from sklearn.model_selection import train_test_split

y = iris_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 135)


model.fit(X_train, y_train)
model.predict(X_train[0].reshape(1, -1))
y_pred = model.predict(X_test)
model.score(X_test, y_test)

from sklearn.metrics import classification_report
print(classification_report(y_pred = y_pred, y_true = y_test))