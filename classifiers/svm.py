from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import numpy as np
from os import path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt


def get_best_model(models):
    highest = 0
    highest_model = None
    for m in models:
        if m.best_score_ > highest:
            highest_model = m
    return highest_model

def svm(X_train, X_test, y_train, y_test):
    if path.exists("./classifiers/svm_params.json"):
        # load params from file
        with open("./classifiers/svm_params.json", "r") as file:
            params = json.load(file)

        try:
            _ = params['coef0']
        except KeyError:
            params['coef0'] = 0.0

        try:
            _ = params['degree']
        except KeyError:
            params['degree'] = 3

        try:
            _ = params['gamma']
        except KeyError:
            params['gamma'] = 'scale'

        model = SVC(C=params['C'],
                    kernel=params['kernel'],
                    coef0=params['coef0'],
                    degree=params['degree'],
                    gamma=params['gamma'])
        model.fit(X_train, y_train.ravel())
    else:
        print("FINDING BEST PARAMETERS.....\n")

        linear_param = {'C': np.logspace(-2, 10, 5)}
        linear_classifier = SVC(kernel='linear', C=1.0)
        linear_grid = GridSearchCV(estimator=linear_classifier,
                                   param_grid=linear_param,
                                   scoring='accuracy',
                                   cv=5,
                                   n_jobs=-1)
        linear_grid.fit(X_train, y_train.ravel())

        poly_param = {
            'C': np.logspace(-2, 10, 5),
            'coef0': np.arange(0.0, 5.0, 1).tolist(),
            'degree': np.arange(0, 10, 2).tolist()
        }
        poly_classifier = SVC(kernel='linear', C=1.0, coef0=0.0, degree=3)
        poly_grid = GridSearchCV(estimator=poly_classifier,
                                 param_grid=poly_param,
                                 scoring='accuracy',
                                 cv=5,
                                 n_jobs=-1)
        poly_grid.fit(X_train, y_train.ravel())

        rbf_param = {
            'C': np.logspace(-2, 10, 5),
            'gamma': np.logspace(-9, 3, 5)
        }
        rbf_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')
        rbf_grid = GridSearchCV(estimator=rbf_classifier,
                                 param_grid=rbf_param,
                                 scoring='accuracy',
                                 cv=5,
                                 n_jobs=-1)
        rbf_grid.fit(X_train, y_train.ravel())

        model = get_best_model([linear_grid, poly_grid, rbf_grid])
        best_params = model.best_params_
        best_params['kernel'] = model.estimator.__dict__['kernel']
        print(f"Best paraments: {best_params}\n")

        with open("./classifiers/svm_params.json", "w+") as file:
            json.dump(model.best_params_, file)

    pred = model.predict(X_test)

    target_names = ['truthful', 'deceptive']
    print(classification_report(y_test, pred, target_names=target_names))

    plot_confusion_matrix(model, X_test, y_test)
    plt.savefig("./Plots/Plots after tuning/svm_confusion_matrix.png")

    plot_roc_curve(model, X_test, y_test)
    plt.savefig("./Plots/Plots after tuning/svm_roc_auc.png")

    return model
