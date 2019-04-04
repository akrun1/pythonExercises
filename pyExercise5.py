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

from sklearn.model_selection import cross_val_score
model = linear_model.LogisticRegression(penalty = 'l2', dual = False)

scores = cross_val_score(model, dataset.data, dataset.target, cv = 5)
scores.mean(), scores.std() * 2

cross_val_score(model, dataset.data, dataset.target, cv = 5, scoring = 'f1_macro')

###
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# 1
Pipeline([('pca', PCA()), ('classifier', SVC())])

# or 2
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(PCA(), SVC())
pipe.steps[1]
pipe.named_steps["svc"]
pipe.named_steps["svc"].decision_function_shape

###
pipe.fit(X_train, y_train)
pipe.score(X_train, y_train)
pipe.score(X_test, y_test)
model.score(X_test, y_test)


###
param_grid = [
    { 'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma':[0.001, 0.0001],  'kernel': ['rbf']},

]

from sklearn.model_selection import GridSearchCV
model = GridSearchCV(SVC(), param_grid, cv = 5, return_train_score = True)
model.fit(X_train, y_train)

import pandas as pd
pd.DataFrame(model.cv_results_)

##
model = GridSearchCV(pipe, dict(svc__C = [0.1, 10, 100]), cv = 5, return_train_score = True)
model.fit(X_train, y_train)