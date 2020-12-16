from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def gaussian(X_train, X_test, y_train, y_test):
    # train a model
    target_names = ['truthful', 'deceptive']

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # training the model on training set
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    # making predictions on the testing set
    y_pred = gnb.predict(X_test)
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

    # Visualize confusion matrix
    plot_confusion_matrix(gnb, X_test, y_test)

    # Visualize ROC curve
    plot_roc_curve(gnb, X_test, y_test)

    return gnb
