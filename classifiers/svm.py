from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np
import os.path
from os import path
import json

def svm(X_train, X_test, y_train, y_test):
    if path.exists("svm_params.json"):
        # load params from file
        with open("svm_params.json", "r") as file:
            params = json.load(file)

        model = SVC(C=params['C'], kernel=params['kernel'], coef0=params['coef0'], degree=params['degree'], gamma=params['gamma'])
        model.fit(X_train, y_train.ravel())
    else:
        print("FINDING BEST PARAMETERS.....\n\n")
        param = {
            'kernel': ['linear', 'poly', 'rbf'],

            # used for linear, polynomial, rbf kernel
            'C': np.logspace(-2, 10, 5),

            # used for polynomial kernel
            'coef0': np.arange( 0.0, 5.0, 1 ).tolist(),

            # used for polynomial kernel (ignored by other kernels)
            'degree': np.arange( 0, 10, 2).tolist(),

            # used for polynomial, rbf kernel
            'gamma': np.logspace(-9, 3, 13)
        }

        svm_classifier = SVC(C=1.0, kernel='linear', coef0=0.0, degree=3, gamma='scale')
        model = GridSearchCV(estimator=svm_classifier,
                             param_grid=param,
                             scoring='accuracy',
                             cv=5,
                             n_jobs=-1)

        # Create and store optimal paraments in a json file ("svm_params.json")
        model.fit(X_train, y_train.ravel())
        print(f"Best paraments: {model.best_params_}")

        with open("svm_params.json", "w") as file:
            json.dump(model.best_params_, file)


    pred = model.predict(X_test)

    target_names = ['truthful', 'deceptive']
    print(classification_report(y_test, pred, target_names=target_names))

    return model
