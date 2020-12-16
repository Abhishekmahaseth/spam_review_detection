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

# this get featuresd for frequency of words
def calculate_freq():
    # get features
    # for calculate_pos_freq
    args.dataset = "deceptive-opinion.csv"
    data = pd.read_csv(args.dataset)
    X = data.iloc[:,5 if args.dataset == "preprocessed.csv" else 4]
    y = data.iloc[:, 0]
    if args.classifier.lower() == 'svm':
        pass
    X = calculate_pos_freq(X)
    return X

def calculate_idf(X):
    # get features
    X = calculate_tf_idf(X)
    return X

# get fetures for bag of words
def bag_of_words(X):
    # ngram allows it to take 2 words at a time
    X = calculate_bag_of_words(X, ngram = 3)
    return X

#  runs the gaussian model
def gaussian(X,y):
    # train a model
    target_names = ['truthful', 'deceptive']
    # splitting X and y into training and testing sets
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
    plt.show()

    return X,y
    
# take user input for the type of feature to run
def user_input(val):
    print("Please choose the feature you want to run by entering a number :\n")
    print("1: Bag of Words 2: calculate_tf_idf 3: calculate_pos_freq ")
    val = int(input())
    return val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', action='store', dest='dataset', default='preprocessed.csv', required=False,
                        help='Path to the dataset that is to be used in the training and testing of the model')
    parser.add_argument('-c', '--classifier', action='store', dest='classifier', default='svm', required=False,
                        help='Classification algorithm to be used Ex. svm, naive')
    args = parser.parse_args()
    data = pd.read_csv(args.dataset)

    print(data.iloc[0]['preprocessed_text'])  # data.iloc[0][5] or data.iloc[0, 5]
    print(data.iloc[0]['deceptive'])  # data.iloc[0][0] or data.iloc[0, 0]
    if args.classifier.lower() == 'svm':
        X = data.iloc[:, 5 if args.dataset == 'preprocessed.csv' else 4]
    y = data.iloc[:, 0]

    input = user_input(input)
    # print("input:",input)
    if input == 1:
        bag = bag_of_words(X)
        gaussian(bag,y)
    elif input == 2:
        tf =  calculate_idf(X)
        gaussian(tf,y)
    elif input == 3:
        freq = calculate_freq()
        print(freq)
        gaussian(freq,y)
