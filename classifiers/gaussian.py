import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV


def gaussian(X_train, X_test, y_train, y_test):
    # train a model
    target_names = ['truthful', 'deceptive']

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # training the model on training set
    gnb = GaussianNB()

    # making predictions on the testing set
    # y_pred = gnb.predict(X_test)

    # params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
    params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}

    gs_NB = GridSearchCV(estimator=gnb,
                     param_grid=params_NB,
                     cv=15,   # use any cross validation technique
                     verbose=2,
                     scoring='accuracy')

    gs_NB.fit(X_train, y_train)

    # prints the var_smoothing value used 
    print("\n\n",gs_NB.best_params_)
    print("\n")



    y_pred = gs_NB.predict(X_test)
    # this provides Precision, Recall, Accuracy, F1-score
    p, r, f, s = score(y_test, y_pred,average=None)
    print("Precision:", p)
    print("Recall:", r)
    print("F1:", f)
    print("Score:", s)
    # this prints Precision, Recall, Accuracy, F1-score
    print(classification_report(y_test, y_pred, target_names=target_names))

    # comparing actual response values (y_test) with predicted response values (y_pred)
    print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
    print('Test Accuracy : %.3f'%gs_NB.score(X_test, y_test)) ## Score method also evaluates accuracy for classification models.
    print('Training Accuracy : %.3f'%gs_NB.score(X_train, y_train))

    # Visualize confusion matrix
    plot_confusion_matrix(gs_NB, X_test, y_test)

    # Visualize ROC curve
    plot_roc_curve(gs_NB, X_test, y_test)

    # results_NB = pd.DataFrame(gs_NB.cv_results_['params'])
    # results_NB['test_score'] = gs_NB.cv_results_['mean_test_score']
    # plt.plot(results_NB['var_smoothing'], results_NB['test_score'], marker = '.')
    # plt.xlabel('Var. Smoothing')
    # plt.ylabel("Mean CV Score")
    # plt.title("NB Performance Comparison")
    # plt.show()
    return gs_NB
