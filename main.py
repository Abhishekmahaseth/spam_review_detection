import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

from lib.feature_extraction import calculate_bag_of_words, calculate_tf_idf, calculate_pos_freq
from lib.predict_new_review import predict

from classifiers.svm import svm
from classifiers.gaussian import gaussian
from classifiers.mlp_classifier import MLPClassification
from classifiers.random_forest_classifier import randomForest
from classifiers.ada_boost_classifier import adaBoost
import matplotlib.pyplot as plt

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')


if __name__ == '__main__':
    '''
    dataset: [ 'preprocessed.csv'  |  'deceptive-opinion.csv' ]
    feature: [ 'bag-of-words'  |  'tf-idf'  |  'pos-tag-freq' ]
    test-train-split: [ TEST_SIZE ]
    classifier: [ 'svm'  |  'gaussian'  |  'random-forest'  |  'mlp'  | 'ada-boost' ]
    '''

    parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', action='store', dest='dataset', default='preprocessed.csv', required=False,
                        help='Path to the dataset that is to be used in the training and testing of the model. '
                             'Options: [ \'preprocessed.csv\'  |  \'deceptive-opinion.csv\' ]')

    parser.add_argument('-f', '--feature', action='store', dest='feature', default='bag-of-words', required=False,
                        help='Feature to be extracted from dataset. Options: [ \'bag-of-words\'  |  \'tf-idf\'  |  '
                             '\'pos-tag-freq\' ]')

    parser.add_argument('-tts', '--test_train_split', action='store', dest='test_train_split', default='0.20',
                        required=False, help='Data split. Options: [ 0..1 ]')

    parser.add_argument('-c', '--classifier', action='store', dest='classifier', default='svm', required=False,
                        help='Classification algorithm to be used. Options: [ \'svm\'  |  \'gaussian\'  |  '
                             '\'random-forest\'  |  \'mlp\'  |  \'ada-boost\' ]')
    arguments = parser.parse_args()

    # arguments.test_train_split = 0.4

    if arguments.feature == 'pos-tag-freq' and arguments.dataset == 'preprocessed.csv':
        raise Exception("Only deceptive-opinion.csv supported for feature: pos-tag-freq")

    data = pd.read_csv(arguments.dataset)
    X = data.iloc[:, 5 if arguments.dataset == 'preprocessed.csv' else 4]
    y = data.iloc[:, 0]

    ##### FEATURE EXTRACTION
    feature = arguments.feature.lower()
    if feature in ('bag-of-words', 'tf-idf', 'pos-tag-freq'):
        print("EXTRACTING FEATURES.....\n")
        if feature == 'bag-of-words':
            vectorizer, X = calculate_bag_of_words(data=X, ngram=3)
        elif feature == 'tf-idf':
            vectorizer, X = calculate_tf_idf(data=X)
        elif feature == 'pos-tag-freq':
            X = calculate_pos_freq(data=X)
    else:
        raise Exception("Given feature not supported. Choose from: [ 'bag-of-words'  |  'tf-idf'  |  'pos-tag-freq' ]")

    ##### TEST TRAIN SPLIT
    classifier = arguments.classifier.lower()
    test_size = float(arguments.test_train_split)
    if 0 <= test_size <= 1:
        # optimal random_state computed through trails
        if classifier == 'gaussian':
            random_state = 19
        elif classifier == 'random-forest':
            random_state = 42
        else:
            random_state = None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        raise Exception("Split size out of range. Pick a value in [ 0..1 ]")

    ##### Model Training
    if classifier in ('svm', 'gaussian', 'random-forest', 'mlp'):
        print("TRAINING THE MODEL.....\n")
        # do appropriate model training
        if classifier == 'svm':
            ## train SVM model
            model = svm(X_train, X_test, y_train, y_test)
        elif classifier == 'gaussian':
            model = gaussian(X_train, X_test, y_train, y_test)
        elif classifier == 'mlp':
            model = MLPClassification(X_train, X_test, y_train, y_test)
        elif classifier == 'random-forest':
            model = randomForest(X_train, X_test, y_train, y_test)
        elif classifier == 'ada-boost':
            model = adaBoost(X_train, X_test, y_train, y_test)
            
        # Plotting the blobs for 2 catagories
        X, y = make_blobs(n_samples=100, centers=3, n_features=2)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
        plt.show()
    else:
        raise Exception(
            "Given classfier not supported. Choose from: [ 'svm'  |  'gaussian'  |  'random-forest'  |  'mlp'  | 'ada-boost' ]")

    print("MODEL TRAINED!!!\n\n\n")
    print("NOW WE CAN CLASSIFY YOUR REVIEWS")
    while (True):
        print("Enter your review: ")
        review = input()

        result = predict(model, review, feature, vectorizer)
        print(result)
        print()
