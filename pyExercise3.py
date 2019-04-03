import pandas as pd
import numpy as np

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
        current_prediction = np.dot(X, weights)
        gradient = np.dot(X.T, (current_prediction - y)) / y.size
        weights -= learning_rate * gradient


def predict_prob(X):
    global weights

    if fit_intercept:
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    return np.dot(X, weights)


###
from sklearn.datasets import load_boston
boston_data = load_boston()

X = boston_data.data
y = boston_data.target

X_df = pd.DataFrame(X, columns = boston_data.feature_names)
print(boston_data.DESCR)


###
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(y)

###
from sklearn.metrics import mean_squared_error
y_pred = [0] * 506
mean_squared_error(y, y_pred)
sns.jointplot(X[:, 5], y)

def manual_model(house):
    return (house[5] - 4) * 10

y_pred = [manual_model(x) for x in X]
mean_squared_error(y, y_pred)
sns.jointplot(X[:, 5], y)

###
from sklearn.linear_model import LinearRegression
model = LinearRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
model.fit(X_train, y_train)
y_pred_ml = model.predict(X_test)
mean_squared_error(y_test, y_pred_ml)

###
model_2 = LinearRegression(normalize = True)
model_2.fit(X_train, y_train)
y_pred_ml2 = model.predict(X_test)
mean_squared_error(y_test, y_pred_ml2)