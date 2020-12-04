import argparse
import pandas as pd
from feature_extraction import *

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

    ## get features
    X = calculate_bag_of_words(X)
    print(X)

    ## split the data

    ## train a model
