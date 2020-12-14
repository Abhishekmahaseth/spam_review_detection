import argparse
import pandas as pd
from feature_extraction import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

def training_GaussianModel(X,y):
    # get features
    # ngram allows it to take 2 words at a time
    X = calculate_bag_of_words(X, ngram = 2)
    # X = calculate_tf_idf(X)
    # X = calculate_pos_freq(X)
    print(X)

    # train a model
    target_names = ['truthful', 'deceptive']
    # splitting X and y into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=19)
    from sklearn.preprocessing import StandardScaler
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

    # Plotting the blobs for 2 catagories
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=100, centers=3, n_features=2)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');
    # plt.show()


    return X,y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', action='store', dest='dataset', default='preprocessed.csv', required=False,
                        help='Path to the dataset that is to be used in the training and testing of the model')
    parser.add_argument('-c', '--classifier', action='store', dest='classifier', default='svm', required=False,
                        help='Classification algorithm to be used Ex. svm, naive')
    args = parser.parse_args()

    data = pd.read_csv(args.dataset)

    # for calculate_pos_freq
    # args.dataset = "deceptive-opinion.csv"
    # data = pd.read_csv(args.dataset)
    # X = data.iloc[:,5 if args.dataset == "preprocessed.csv" else 4]
    # y = data.iloc[:, 0]
    # if args.classifier.lower() == 'svm':
    #     pass

    print(data.iloc[0]['preprocessed_text'])  # data.iloc[0][5] or data.iloc[0, 5]
    print(data.iloc[0]['deceptive'])  # data.iloc[0][0] or data.iloc[0, 0]
    if args.classifier.lower() == 'svm':
        X = data.iloc[:, 5 if args.dataset == 'preprocessed.csv' else 4]
    y = data.iloc[:, 0]

    training_GaussianModel(X,y)
