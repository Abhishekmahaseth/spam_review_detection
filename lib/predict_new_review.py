from .feature_extraction import calculate_bag_of_words, calculate_tf_idf, calculate_pos_freq

def predict(model, review, feature):
    if feature == 'bag-of-words':
        X = calculate_bag_of_words(data=X, ngram=2)
    elif feature == 'tf-idf':
        X = calculate_tf_idf(data=X)
    elif feature == 'pos-tag-freq':
        X = calculate_pos_freq(data=X)

    y_review = model.predict(X)
    #######
    # TO DO: Check if y_review needs to converted to be in ['truthful', 'deceptive']
    #######
