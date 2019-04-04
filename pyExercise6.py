from sklearn.datasets import load_boston
boston_data = load_boston()

X = boston_data.data
y = boston_data.target

from sklearn.linear_model import LinearRegression
model = LinearRegression()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
model.fit(X_train, y_train)
model.score(X_train, y_train)


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv = 5)
scores

from sklearn.metrics import explained_variance_score
explained_variance_score(y_test, model.predict(X_test))

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, model.predict(X_test))

from sklearn.metrics import median_absolute_error
median_absolute_error(y_test, model.predict(X_test))

from sklearn.metrics import r2_score
r2_score(y_test, model.predict(X_test))


from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree = 3), LinearRegression())
pipe.fit(X_train, y_train)

pipe.score(X_train, y_train)
pipe.score(X_test, y_test)
pipe.steps[2][1].coef_
pipe.steps[2][1].coef_.max(), pipe.steps[2][1].coef_.min(), pipe.steps[2][1].coef_.std()

##
from sklearn.linear_model import Ridge
pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree = 3), Ridge())
pipe.fit(X_train, y_train)

pipe.score(X_train, y_train)
pipe.score(X_test, y_test)
pipe.steps[2][1].coef_.max(), pipe.steps[2][1].coef_.min(), pipe.steps[2][1].coef_.std()

##
from sklearn.linear_model import Lasso
pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree = 3), Lasso())
pipe.fit(X_train, y_train)

pipe.score(X_train, y_train)
pipe.score(X_test, y_test)
pipe.steps[2][1].coef_.max(), pipe.steps[2][1].coef_.min(), pipe.steps[2][1].coef_.std()

##
from sklearn.linear_model import ElasticNet
pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree = 3), ElasticNet())
pipe.fit(X_train, y_train)

pipe.score(X_train, y_train)
pipe.score(X_test, y_test)
pipe.steps[2][1].coef_.max(), pipe.steps[2][1].coef_.min(), pipe.steps[2][1].coef_.std()



####
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import linear_model
dataset = datasets.load_iris()


dataset.data.shape
dataset.target.shape

holdout_percent = 0.4
X_train, X_test, y_train, y_test = train_test_split(dataset.data, \
                                                    dataset.target, test_size = holdout_percent, random_state = 42)

model = linear_model.LogisticRegression(penalty = 'l2', dual = False, tol = 0.0001, \
                                        C = 1.0, fit_intercept = True).fit(X_train, y_train)

model.score(X_train, y_train)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, model.predict(X_test))

from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(y_test, model.predict(X_test))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, model.predict(X_test))

###

